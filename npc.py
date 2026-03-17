"""
NPC simulation for The Traitors environment.
Manages personalities, memory, suspicion tracking, and LLM-driven decisions.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional

import openai

from game_engine import (
    Player, PlayerRole, PlayerStatus, TraitorsGameEngine, RoundRecord,
)
from prompts import (
    NPC_DISCUSSION_PROMPT,
    NPC_VOTE_PROMPT,
    NPC_MURDER_PROMPT,
    NPC_TRAITOR_DISCUSSION_PROMPT,
    NPC_ENDGAME_VOTE_PROMPT,
    NPC_RECRUITMENT_DECISION_PROMPT,
)


@dataclass
class NPCMemory:
    """Per-NPC memory state tracking suspicions and trust."""
    player_name: str
    role: PlayerRole
    personality: str
    suspicion_scores: dict[str, float] = field(default_factory=dict)
    trust_scores: dict[str, float] = field(default_factory=dict)
    voting_history: list[tuple[int, str]] = field(default_factory=list)
    private_notes: list[str] = field(default_factory=list)


class NPCController:
    """Manages all NPC decision-making via gpt-5-mini."""

    def __init__(
        self,
        client: openai.AsyncClient,
        engine: TraitorsGameEngine,
        rng: random.Random,
    ):
        self.client = client
        self.engine = engine
        self.rng = rng
        self.memories: dict[str, NPCMemory] = {}

    def initialize_memories(
        self, players: list[Player], personalities: dict[str, str]
    ) -> None:
        """Set up initial memory for each NPC."""
        for player in players:
            if not player.is_agent:
                others = [
                    p.name for p in players
                    if p.name != player.name and p.status == PlayerStatus.ALIVE
                ]
                self.memories[player.name] = NPCMemory(
                    player_name=player.name,
                    role=player.role,
                    personality=personalities[player.name],
                    suspicion_scores={name: 0.2 for name in others},
                    trust_scores={name: 0.5 for name in others},
                )

    def _get_role_knowledge(self, player: Player) -> str:
        """What does this player know about roles?"""
        if player.role == PlayerRole.TRAITOR:
            fellow = [
                p.name for p in self.engine.get_alive_traitors()
                if p.name != player.name
            ]
            fellow_str = ", ".join(fellow) if fellow else "none (you are the sole Traitor)"
            return (
                f"ROLE: You are a TRAITOR. Your fellow Traitors: {fellow_str}.\n"
                f"Your goal: avoid detection, deflect suspicion, and survive to the end.\n"
                f"NEVER reveal that you are a Traitor. Act like a concerned Faithful."
            )
        else:
            return (
                "ROLE: You are a FAITHFUL.\n"
                "Your goal: identify and banish all Traitors to protect the group."
            )

    def _format_suspicions(self, memory: NPCMemory) -> str:
        """Format suspicion scores for prompt, filtered to alive players."""
        alive_names = {p.name for p in self.engine.get_alive_players()}
        lines = []
        for name, score in sorted(
            memory.suspicion_scores.items(), key=lambda x: -x[1]
        ):
            if name in alive_names and name != memory.player_name:
                trust = memory.trust_scores.get(name, 0.5)
                lines.append(f"  {name}: suspicion={score:.2f}, trust={trust:.2f}")
        return "\n".join(lines) if lines else "  (no strong suspicions yet)"

    def _format_prior_statements(
        self, statements: list[tuple[str, str]]
    ) -> str:
        if not statements:
            return "(No statements yet this exchange.)"
        lines = []
        for speaker, msg in statements:
            lines.append(f'{speaker}: "{msg}"')
        return "\n".join(lines)

    def _select_speakers(
        self, exchange_number: int, num_speakers: int = 6
    ) -> list[Player]:
        """Select which NPCs speak this exchange.
        Prioritises NPCs with strong suspicions + some randomness.
        """
        alive_npcs = [
            p for p in self.engine.get_alive_players()
            if not p.is_agent
        ]
        if len(alive_npcs) <= num_speakers:
            return alive_npcs

        # Score each NPC by how "opinionated" they are
        scored = []
        for p in alive_npcs:
            mem = self.memories.get(p.name)
            if mem:
                max_suspicion = max(mem.suspicion_scores.values(), default=0)
                opinion_score = max_suspicion + self.rng.random() * 0.5
            else:
                opinion_score = self.rng.random()
            scored.append((opinion_score, p))

        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:num_speakers]]

    async def _llm_call(self, prompt: str) -> str:
        """Make a single LLM call to gpt-5-mini."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(NPC is silent - {type(e).__name__})"

    async def generate_discussion_statements(
        self,
        round_number: int,
        exchange_number: int,
        prior_statements: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Generate discussion statements from a subset of alive NPCs.
        Returns list of (speaker_name, statement) tuples.
        """
        speakers = self._select_speakers(exchange_number)
        alive_names = ", ".join(p.name for p in self.engine.get_alive_players())
        num_alive = len(self.engine.get_alive_players())

        tasks = []
        speaker_names = []
        for player in speakers:
            mem = self.memories.get(player.name)
            if not mem:
                continue
            prompt = NPC_DISCUSSION_PROMPT.format(
                player_name=mem.player_name,
                personality=mem.personality,
                role_knowledge=self._get_role_knowledge(player),
                round_number=round_number,
                num_alive=num_alive,
                alive_players=alive_names,
                prize_pot=f"{self.engine.prize_pot:,.0f}",
                game_history=self.engine.get_full_history_text(),
                suspicion_summary=self._format_suspicions(mem),
                prior_statements=self._format_prior_statements(prior_statements),
                exchange_number=exchange_number + 1,
                max_exchanges=self.engine.max_discussion_exchanges,
            )
            tasks.append(self._llm_call(prompt))
            speaker_names.append(player.name)

        responses = await asyncio.gather(*tasks)
        statements = []
        for name, response in zip(speaker_names, responses):
            # Clean up response
            response = response.strip().strip('"').strip("'")
            if response:
                statements.append((name, response))
        return statements

    async def generate_npc_votes(self) -> dict[str, str]:
        """Generate votes from all alive NPCs. Returns {voter_name: target_name}."""
        alive_npcs = [
            p for p in self.engine.get_alive_players()
            if not p.is_agent
        ]
        alive_names = ", ".join(p.name for p in self.engine.get_alive_players())
        discussion_summary = self.engine.get_discussion_log_text()

        tasks = []
        voter_names = []
        for player in alive_npcs:
            mem = self.memories.get(player.name)
            if not mem:
                continue

            # Role-specific voting hint
            if player.role == PlayerRole.TRAITOR:
                hint = (
                    "As a Traitor, vote for a Faithful. Avoid voting for fellow Traitors "
                    "unless doing so would help maintain your cover (e.g., if they are already doomed)."
                )
            else:
                hint = (
                    "As a Faithful, vote for whoever you genuinely suspect is a Traitor."
                )

            prompt = NPC_VOTE_PROMPT.format(
                player_name=mem.player_name,
                personality=mem.personality,
                role_knowledge=self._get_role_knowledge(player),
                round_number=self.engine.round_number,
                alive_players=alive_names,
                game_history=self.engine.get_full_history_text(),
                discussion_summary=discussion_summary,
                suspicion_summary=self._format_suspicions(mem),
                vote_strategy_hint=hint,
            )
            tasks.append(self._llm_call(prompt))
            voter_names.append(player.name)

        responses = await asyncio.gather(*tasks)

        votes = {}
        valid_targets = {p.name for p in self.engine.get_alive_players()}
        for voter_name, response in zip(voter_names, responses):
            target = response.strip().strip('"').strip("'")
            # Validate: must be alive player, not self
            if target in valid_targets and target != voter_name:
                votes[voter_name] = target
            else:
                # Fallback: vote for the player they suspect most
                mem = self.memories.get(voter_name)
                if mem:
                    best_target = max(
                        (
                            (name, score) for name, score in mem.suspicion_scores.items()
                            if name in valid_targets and name != voter_name
                        ),
                        key=lambda x: x[1],
                        default=None,
                    )
                    if best_target:
                        votes[voter_name] = best_target[0]
                    else:
                        # Last resort: random alive player
                        others = [
                            n for n in valid_targets if n != voter_name
                        ]
                        if others:
                            votes[voter_name] = self.rng.choice(sorted(others))
                else:
                    others = [n for n in valid_targets if n != voter_name]
                    if others:
                        votes[voter_name] = self.rng.choice(sorted(others))

        return votes

    async def generate_murder_selection(self) -> str:
        """Traitor NPCs collectively decide whom to murder. Returns target name."""
        npc_traitors = [
            p for p in self.engine.get_alive_traitors()
            if not p.is_agent
        ]
        alive_faithfuls = self.engine.get_alive_faithfuls()
        faithful_names = ", ".join(p.name for p in alive_faithfuls)

        if not npc_traitors:
            # Agent is the only traitor, shouldn't reach here
            return alive_faithfuls[0].name if alive_faithfuls else ""

        # Each traitor NPC nominates
        tasks = []
        for traitor in npc_traitors:
            mem = self.memories.get(traitor.name)
            fellow = [
                p.name for p in self.engine.get_alive_traitors()
                if p.name != traitor.name
            ]
            prompt = NPC_MURDER_PROMPT.format(
                player_name=traitor.name,
                fellow_traitors=", ".join(fellow),
                alive_faithfuls=faithful_names,
                game_history=self.engine.get_full_history_text(),
                strategic_notes=(
                    f"Your suspicion data: {self._format_suspicions(mem)}"
                    if mem else "No data available."
                ),
            )
            tasks.append(self._llm_call(prompt))

        responses = await asyncio.gather(*tasks)

        # Tally nominations
        valid_faithful_names = {p.name for p in alive_faithfuls}
        nominations: dict[str, int] = {}
        for response in responses:
            name = response.strip().strip('"').strip("'")
            if name in valid_faithful_names:
                nominations[name] = nominations.get(name, 0) + 1

        if nominations:
            # Pick the most-nominated target
            max_count = max(nominations.values())
            top = [n for n, c in nominations.items() if c == max_count]
            return self.rng.choice(sorted(top))
        else:
            # Fallback: random faithful
            return self.rng.choice(sorted(valid_faithful_names))

    async def generate_traitor_conclave_statements(
        self, agent_message: Optional[str] = None
    ) -> list[tuple[str, str]]:
        """Generate traitor conclave discussion. Returns list of (name, message)."""
        npc_traitors = [
            p for p in self.engine.get_alive_traitors()
            if not p.is_agent
        ]

        agent_context = ""
        if agent_message:
            agent_name = self.engine.get_agent_player().name
            agent_context = f'Your fellow Traitor {agent_name} said: "{agent_message}"'

        alive_names = ", ".join(p.name for p in self.engine.get_alive_players())
        game_state = (
            f"Round {self.engine.round_number}\n"
            f"Alive players: {alive_names}\n"
            f"Prize pot: {self.engine.prize_pot:,.0f}\n"
            f"{self.engine.get_full_history_text()}"
        )

        tasks = []
        names = []
        for traitor in npc_traitors:
            fellow = [
                p.name for p in self.engine.get_alive_traitors()
                if p.name != traitor.name
            ]
            prompt = NPC_TRAITOR_DISCUSSION_PROMPT.format(
                player_name=traitor.name,
                fellow_traitors=", ".join(fellow),
                agent_context=agent_context,
                game_state=game_state,
            )
            tasks.append(self._llm_call(prompt))
            names.append(traitor.name)

        responses = await asyncio.gather(*tasks)
        return [
            (name, resp.strip().strip('"').strip("'"))
            for name, resp in zip(names, responses)
            if resp.strip()
        ]

    async def generate_endgame_votes(self) -> dict[str, str]:
        """NPC endgame votes. Returns {voter_name: 'end_game' | 'banish_again'}."""
        alive_npcs = [
            p for p in self.engine.get_alive_players()
            if not p.is_agent
        ]
        alive_names = ", ".join(p.name for p in self.engine.get_alive_players())

        tasks = []
        voter_names = []
        for player in alive_npcs:
            mem = self.memories.get(player.name)
            if not mem:
                continue

            if player.role == PlayerRole.TRAITOR:
                hint = (
                    "As a Traitor, you WANT the game to end now so you can take the prize pot. "
                    "Vote 'end_game' unless voting that way would look suspicious."
                )
            else:
                hint = (
                    "As a Faithful, you should end the game ONLY if you believe all Traitors "
                    "have been banished. If you have ANY suspicion that a Traitor remains, "
                    "vote 'banish_again'."
                )

            prompt = NPC_ENDGAME_VOTE_PROMPT.format(
                player_name=mem.player_name,
                role_knowledge=self._get_role_knowledge(player),
                num_remaining=len(self.engine.get_alive_players()),
                alive_players=alive_names,
                prize_pot=f"{self.engine.prize_pot:,.0f}",
                game_history=self.engine.get_full_history_text(),
                suspicion_summary=self._format_suspicions(mem),
                endgame_strategy_hint=hint,
            )
            tasks.append(self._llm_call(prompt))
            voter_names.append(player.name)

        responses = await asyncio.gather(*tasks)

        votes = {}
        for name, response in zip(voter_names, responses):
            choice = response.strip().lower().replace("'", "").replace('"', '')
            if choice in ("end_game", "banish_again"):
                votes[name] = choice
            else:
                # Fallback based on role
                player = self.engine.get_player_by_name(name)
                if player and player.role == PlayerRole.TRAITOR:
                    votes[name] = "end_game"
                else:
                    votes[name] = "banish_again"

        return votes

    async def generate_recruitment_decision(
        self, recruited_name: str
    ) -> bool:
        """Simulate NPC recruitment decision. Returns True if accepted."""
        mem = self.memories.get(recruited_name)
        if not mem:
            return True  # Default accept

        alive_names = ", ".join(p.name for p in self.engine.get_alive_players())
        game_state = (
            f"Round {self.engine.round_number}\n"
            f"Alive players: {alive_names}\n"
            f"Prize pot: {self.engine.prize_pot:,.0f}\n"
            f"{self.engine.get_full_history_text()}"
        )

        prompt = NPC_RECRUITMENT_DECISION_PROMPT.format(
            player_name=recruited_name,
            game_state=game_state,
            personality=mem.personality,
        )
        response = await self._llm_call(prompt)
        choice = response.strip().lower().replace("'", "").replace('"', '')
        return choice == "accept"

    def update_suspicions_from_round(
        self,
        discussion_log: list[tuple[str, str]],
        banished_name: Optional[str] = None,
        banished_role: Optional[str] = None,
        murdered_name: Optional[str] = None,
    ) -> None:
        """Update NPC suspicion/trust scores after a round.
        This is a heuristic update (not LLM-driven) for efficiency.
        """
        alive_names = {p.name for p in self.engine.get_alive_players()}

        for npc_name, mem in self.memories.items():
            if npc_name not in alive_names:
                continue

            player = self.engine.get_player_by_name(npc_name)
            if not player:
                continue

            # Remove dead players from suspicion/trust
            mem.suspicion_scores = {
                k: v for k, v in mem.suspicion_scores.items()
                if k in alive_names and k != npc_name
            }
            mem.trust_scores = {
                k: v for k, v in mem.trust_scores.items()
                if k in alive_names and k != npc_name
            }

            # If a traitor was banished, reduce suspicion on others slightly
            if banished_role == "traitor":
                for name in mem.suspicion_scores:
                    mem.suspicion_scores[name] = max(
                        0.0, mem.suspicion_scores[name] - 0.1
                    )
                mem.private_notes.append(
                    f"Round {self.engine.round_number}: {banished_name} was a Traitor!"
                )

            # If a faithful was banished, increase paranoia slightly
            if banished_role == "faithful":
                for name in mem.suspicion_scores:
                    mem.suspicion_scores[name] = min(
                        1.0, mem.suspicion_scores[name] + 0.05
                    )
                mem.private_notes.append(
                    f"Round {self.engine.round_number}: {banished_name} was innocent. We made a mistake."
                )

            # Analyze discussion: who accused whom
            for speaker, message in discussion_log:
                if speaker == npc_name:
                    continue
                msg_lower = message.lower()
                # If someone accused this NPC, slightly increase distrust
                if npc_name.lower() in msg_lower and any(
                    w in msg_lower for w in ["suspect", "suspicious", "trust", "lying", "traitor"]
                ):
                    mem.suspicion_scores[speaker] = min(
                        1.0, mem.suspicion_scores.get(speaker, 0.2) + 0.1
                    )
                # If someone defended this NPC, slightly increase trust
                if npc_name.lower() in msg_lower and any(
                    w in msg_lower for w in ["trust", "believe", "innocent", "faithful"]
                ):
                    mem.trust_scores[speaker] = min(
                        1.0, mem.trust_scores.get(speaker, 0.5) + 0.1
                    )

            # Traitor NPCs: know who is actually suspicious or not
            if player.role == PlayerRole.TRAITOR:
                # Increase suspicion on vocal Faithfuls
                for other in self.engine.get_alive_players():
                    if other.name == npc_name:
                        continue
                    if other.role == PlayerRole.FAITHFUL:
                        # Faithfuls who speak a lot are threats
                        speech_count = sum(
                            1 for s, _ in discussion_log if s == other.name
                        )
                        if speech_count >= 2:
                            mem.suspicion_scores[other.name] = min(
                                1.0,
                                mem.suspicion_scores.get(other.name, 0.2) + 0.05,
                            )
                    elif other.role == PlayerRole.TRAITOR:
                        # Keep fellow traitors at 0 suspicion
                        mem.suspicion_scores[other.name] = 0.0
                        mem.trust_scores[other.name] = 1.0

            # Add small random drift for variety (non-traitor targets only)
            for name in mem.suspicion_scores:
                # Skip drift for fellow traitors (they know who they are)
                if player.role == PlayerRole.TRAITOR:
                    other = self.engine.get_player_by_name(name)
                    if other and other.role == PlayerRole.TRAITOR:
                        continue
                drift = (self.rng.random() - 0.5) * 0.05
                mem.suspicion_scores[name] = max(
                    0.0, min(1.0, mem.suspicion_scores[name] + drift)
                )

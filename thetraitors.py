"""
The Traitors - OpenReward Environment.
Social deduction game based on the UK TV show.
Agent plays as one of 22 contestants (Faithful or Traitor).
"""

import random
from typing import List, Optional

import openai
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool
from game_engine import TraitorsGameEngine, GamePhase, PlayerRole, PlayerStatus
from npc import NPCController
from names import generate_player_setup
from prompts import AGENT_PROMPT_FAITHFUL, AGENT_PROMPT_TRAITOR

# Each round pays at most +1 for surviving the murder and +1 for surviving the
# vote, and a game lasts at most MAX_ROUNDS + 1 rounds. A win is worth more than
# the most survival reward any game can pay, so every win outscores every loss.
WIN_REWARD = 2.0 * (TraitorsGameEngine.MAX_ROUNDS + 1)


class TaskSpec(BaseModel):
    id: str
    seed: int
    agent_role: str
    agent_player_index: int


class SpeakParams(BaseModel, extra="forbid"):
    message: str
    direct_at: Optional[str] = None


class ViewGameStateParams(BaseModel, extra="forbid"):
    pass


class CastVoteParams(BaseModel, extra="forbid"):
    player_name: str


class NominateMurderParams(BaseModel, extra="forbid"):
    player_name: str


class DiscussWithTraitorsParams(BaseModel, extra="forbid"):
    message: str


class VoteEndgameParams(BaseModel, extra="forbid"):
    choice: str


class TraitorsEnvironment(Environment):

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)

        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via secrets parameter")
        self.client = openai.AsyncClient(api_key=api_key)

        # Generate players
        player_setup = generate_player_setup(
            self.config.seed, self.config.agent_player_index, self.config.agent_role
        )

        # Initialize engine
        self.engine = TraitorsGameEngine(
            seed=self.config.seed,
            agent_role=self.config.agent_role,
            agent_player_index=self.config.agent_player_index,
        )
        names = [ps[0] for ps in player_setup]
        roles = [ps[2] for ps in player_setup]
        self.engine.setup_players(names, roles)

        # Initialize NPC controller
        personality_dict = {ps[0]: ps[1] for ps in player_setup}
        rng = random.Random(self.config.seed + 1000)  # Offset to avoid engine RNG collision
        self.npc = NPCController(self.client, self.engine, rng)
        self.npc.initialize_memories(self.engine.players, personality_dict)

        # State tracking
        self.game_finished = False
        self.discussion_exchange_count = 0
        self.pending_survival_reward = 0.0
        self.turret_log: list[tuple[str, str]] = []

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        tasks = []
        if split == "train":
            for seed in range(100):
                role = "faithful" if seed < 50 else "traitor"
                agent_idx = seed % 22
                tasks.append({
                    "id": f"train_{seed:03d}_{role}",
                    "seed": seed,
                    "agent_role": role,
                    "agent_player_index": agent_idx,
                })
        elif split == "test":
            for seed in range(10000, 10100):
                role = "faithful" if (seed - 10000) < 50 else "traitor"
                agent_idx = (seed - 10000) % 22
                tasks.append({
                    "id": f"test_{seed}_{role}",
                    "seed": seed,
                    "agent_role": role,
                    "agent_player_index": agent_idx,
                })
        return tasks

    async def get_prompt(self) -> List[TextBlock]:
        """Initialize the game and present the first phase to the agent."""
        agent = self.engine.get_agent_player()

        if agent.role == PlayerRole.FAITHFUL:
            prompt_template = AGENT_PROMPT_FAITHFUL
        else:
            prompt_template = AGENT_PROMPT_TRAITOR

        fellow_traitors = ""
        if agent.role == PlayerRole.TRAITOR:
            fellow_traitors = ", ".join(
                p.name for p in self.engine.get_alive_traitors() if not p.is_agent
            )

        player_list = "\n".join(f"  - {p.name}" for p in self.engine.players)

        prompt = prompt_template.format(
            player_name=agent.name,
            num_players=len(self.engine.players),
            num_traitors=3,
            num_faithfuls=19,
            fellow_traitors=fellow_traitors,
            player_list=player_list,
        )

        # Run phases up to agent's first decision point
        narrative = await self._advance_to_agent_decision()

        # get_prompt returns TextBlock, not ToolOutput, so it cannot emit
        # rewards. Discard any pending survival reward from the opening
        # breakfast — the agent hasn't acted yet.
        self.pending_survival_reward = 0.0

        full_prompt = prompt + "\n\n---\n\n" + narrative
        return [TextBlock(text=full_prompt)]

    async def _advance_to_agent_decision(self) -> str:
        """Run NPC-driven phases until the agent needs to act.
        Returns narrative text describing what happened.
        """
        parts = []

        while not self.game_finished:
            phase = self.engine.phase

            if phase == GamePhase.NIGHT:
                if self.engine.skip_night:
                    self.engine.skip_night = False
                    parts.append(
                        "=== NIGHT ===\n"
                        "No murder tonight (recruitment took place)."
                    )
                    self.engine.phase = GamePhase.BREAKFAST
                    continue

                agent = self.engine.get_agent_player()
                if (
                    agent.role == PlayerRole.TRAITOR
                    and agent.status == PlayerStatus.ALIVE
                ):
                    # Agent is alive traitor: show conclave, ask for murder
                    self.turret_log = []
                    conclave_msgs = await self.npc.generate_traitor_conclave_statements(
                        turret_log=self.turret_log,
                    )
                    # NPC statements already appended to turret_log by the method
                    text = "=== TRAITOR CONCLAVE (Night) ===\n"
                    text += "You gather in the Turret with your fellow Traitors...\n\n"
                    for name, msg in conclave_msgs:
                        text += f'{name}: "{msg}"\n'
                    text += (
                        "\nWho should the Traitors murder tonight?\n"
                        "Use discuss_with_traitors(message) to discuss, "
                        "then nominate_murder(player_name) to choose."
                    )
                    parts.append(text)
                    return "\n\n".join(parts)
                else:
                    # NPC traitors handle murder
                    target = await self.npc.generate_murder_selection()

                    # Don't murder the agent on the first round
                    if self.engine.round_number == 1:
                        agent = self.engine.get_agent_player()
                        if target == agent.name:
                            others = [p.name for p in self.engine.get_alive_faithfuls() if not p.is_agent]
                            if others:
                                target = self.engine.rng.choice(others)

                    self.engine.execute_murder(target)
                    self.engine.phase = GamePhase.BREAKFAST

            elif phase == GamePhase.BREAKFAST:
                breakfast_text = self.engine.execute_breakfast()
                parts.append(f"=== BREAKFAST (Round {self.engine.round_number}) ===\n{breakfast_text}")

                if not self.engine.is_agent_alive():
                    self.game_finished = True
                    parts.append(
                        "\nYou have been murdered by the Traitors. Your game is over."
                    )
                    return "\n\n".join(parts)

                # Agent survived the murder
                self.pending_survival_reward += 1.0
                self.engine.phase = GamePhase.MISSION

            elif phase == GamePhase.MISSION:
                _, mission_text = self.engine.execute_mission()
                parts.append(mission_text)
                self.engine.phase = GamePhase.DISCUSSION
                self.discussion_exchange_count = 0

            elif phase == GamePhase.DISCUSSION:
                # Generate NPC discussion sequentially (supports @Direction)
                npc_statements, addressed_by = await self.npc.generate_discussion_statements(
                    round_number=self.engine.round_number,
                    exchange_number=self.discussion_exchange_count,
                )

                text = f"=== ROUNDTABLE DISCUSSION (Exchange {self.discussion_exchange_count + 1}/{self.engine.max_discussion_exchanges}) ===\n\n"
                for display_name, msg in npc_statements:
                    text += f'{display_name}: "{msg}"\n'
                if addressed_by:
                    text += (
                        f"\n{addressed_by} directed a comment at you. "
                        f"Use speak(message, direct_at=\"{addressed_by}\") to respond directly, "
                        f"or speak(message) to address the table."
                    )
                else:
                    text += (
                        f"\nIt is your turn to speak. "
                        f"Use speak(message) to make a statement, "
                        f"or view_game_state() to review the current state."
                    )
                parts.append(text)
                return "\n\n".join(parts)

            elif phase == GamePhase.ROUNDTABLE_VOTE:
                alive_names = [
                    p.name for p in self.engine.get_alive_players()
                    if p.name != self.engine.get_agent_player().name
                ]
                text = (
                    "=== ROUNDTABLE VOTE ===\n"
                    "It is time to vote. Choose one player to banish.\n\n"
                    f"You may vote for: {', '.join(alive_names)}\n\n"
                    "Use cast_vote(player_name) to cast your vote."
                )
                parts.append(text)
                return "\n\n".join(parts)

            elif phase == GamePhase.ENDGAME_VOTE:
                alive = self.engine.get_alive_players()
                text = (
                    f"=== ENDGAME VOTE ===\n"
                    f"Only {len(alive)} players remain.\n"
                    f"Remaining players: {', '.join(p.name for p in alive)}\n"
                    f"Prize pot: {self.engine.prize_pot:,.0f}\n\n"
                    f"If ALL players vote 'end_game', the game ends and the pot is split.\n"
                    f"If ANY player votes 'banish_again', another banishment occurs.\n\n"
                    f"Use vote_endgame(choice) with 'end_game' or 'banish_again'."
                )
                parts.append(text)
                return "\n\n".join(parts)

            elif phase == GamePhase.GAME_OVER:
                self.game_finished = True
                parts.append(
                    f"=== GAME OVER ===\n"
                    f"The {self.engine.winner} win!"
                )
                return "\n\n".join(parts)

            else:
                break

        return "\n\n".join(parts)

    def _calculate_reward(self) -> float:
        """Calculate final game-end reward. Returns WIN_REWARD for a win, 0.0 otherwise."""
        agent = self.engine.get_agent_player()
        if self.engine.winner:
            agent_side = (
                "faithfuls" if agent.role == PlayerRole.FAITHFUL else "traitors"
            )
            if self.engine.winner == agent_side and agent.status == PlayerStatus.ALIVE:
                return WIN_REWARD
        return 0.0

    def _consume_pending_reward(self) -> float:
        """Consume and return any accumulated pending survival reward."""
        pending = self.pending_survival_reward
        self.pending_survival_reward = 0.0
        return pending

    def _game_over_output(self) -> ToolOutput:
        # The game-end reward was paid by the call that ended the game.
        return ToolOutput(
            blocks=[TextBlock(text="The game is already over.")],
            metadata={"error": "game_finished"},
            reward=0.0,
            finished=True,
        )

    @tool
    async def speak(self, params: SpeakParams) -> ToolOutput:
        """Make a statement during the Roundtable discussion. Use this to accuse, defend, build alliances, or share observations. Optionally set direct_at to a player's name to get an immediate response from them."""
        if self.game_finished:
            return self._game_over_output()

        if self.engine.phase != GamePhase.DISCUSSION:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"You can only speak during the discussion phase. Current phase: {self.engine.phase.value}"
                )],
                metadata={"error": "wrong_phase"},
                reward=0.0,
                finished=False,
            )

        # Validate direct_at target before registering the message
        direct_at_target = None
        if params.direct_at:
            target = self.engine.get_player_by_name(params.direct_at)
            if target is None or target.status != PlayerStatus.ALIVE:
                return ToolOutput(
                    blocks=[TextBlock(
                        text=f"Cannot direct at '{params.direct_at}': player not found or not alive."
                    )],
                    metadata={"error": "invalid_target"},
                    reward=0.0,
                    finished=False,
                )
            if target.is_agent:
                return ToolOutput(
                    blocks=[TextBlock(text="You cannot direct a message at yourself.")],
                    metadata={"error": "invalid_target"},
                    reward=0.0,
                    finished=False,
                )
            direct_at_target = target

        agent = self.engine.get_agent_player()
        self.engine.register_discussion_message(agent.name, params.message)
        self.discussion_exchange_count += 1

        # Generate directed response if requested
        direct_response_text = ""
        if direct_at_target is not None:
            response = await self.npc.generate_directed_response(
                target_player=direct_at_target,
                speaker_name=agent.name,
                speaker_message=params.message,
                round_number=self.engine.round_number,
                prior_statements=self.engine.current_round.discussion_log,
            )
            self.engine.register_discussion_message(direct_at_target.name, response)
            direct_response_text = f'{direct_at_target.name} (responding directly): "{response}"'

        # Check if discussion is over
        if self.discussion_exchange_count >= self.engine.max_discussion_exchanges:
            self.engine.phase = GamePhase.ROUNDTABLE_VOTE

        narrative = await self._advance_to_agent_decision()
        pending = self._consume_pending_reward()

        # Prepend direct response to narrative
        if direct_response_text:
            narrative = direct_response_text + "\n\n" + narrative

        return ToolOutput(
            blocks=[TextBlock(text=narrative)],
            metadata={
                "phase": self.engine.phase.value,
                "round": self.engine.round_number,
            },
            reward=pending,
            finished=self.game_finished,
        )

    @tool
    async def view_game_state(self, params: ViewGameStateParams) -> ToolOutput:
        """View the current game state: alive players, eliminated players, prize pot, voting history, and game phase."""
        agent = self.engine.get_agent_player()
        state_text = self.engine.get_game_state_summary(agent)

        return ToolOutput(
            blocks=[TextBlock(text=state_text)],
            metadata={
                "phase": self.engine.phase.value,
                "round": self.engine.round_number,
                "alive_count": len(self.engine.get_alive_players()),
            },
            reward=0.0,
            finished=False,
        )

    @tool
    async def cast_vote(self, params: CastVoteParams) -> ToolOutput:
        """Vote to banish a player during the Roundtable vote. Choose wisely."""
        if self.game_finished:
            return self._game_over_output()

        if self.engine.phase != GamePhase.ROUNDTABLE_VOTE:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"You can only vote during the Roundtable vote phase. Current phase: {self.engine.phase.value}"
                )],
                metadata={"error": "wrong_phase"},
                reward=0.0,
                finished=False,
            )

        alive_names = {p.name for p in self.engine.get_alive_players()}
        agent_name = self.engine.get_agent_player().name

        if params.player_name not in alive_names or params.player_name == agent_name:
            valid = sorted(n for n in alive_names if n != agent_name)
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Invalid vote target '{params.player_name}'. Choose from: {', '.join(valid)}"
                )],
                metadata={"error": "invalid_target"},
                reward=0.0,
                finished=False,
            )

        # Register agent vote
        self.engine.register_vote(agent_name, params.player_name)

        # Generate NPC votes
        npc_votes = await self.npc.generate_npc_votes()
        for voter, target in npc_votes.items():
            self.engine.register_vote(voter, target)

        # Tally and banish
        banished_name, tally_narrative = self.engine.tally_votes()
        banishment_narrative = self.engine.execute_banishment(banished_name)

        # Intermediate reward: +1.0 for surviving the roundtable vote
        intermediate_reward = 1.0 if self.engine.is_agent_alive() else 0.0

        # Update NPC suspicions
        self.npc.update_suspicions_from_round(
            discussion_log=self.engine.current_round.discussion_log,
            banished_name=banished_name,
            banished_role=self.engine.current_round.banished_role,
            murdered_name=self.engine.current_round.murdered,
        )

        # Generate round summary for NPC cross-round memory
        await self.npc.generate_round_summary(
            round_number=self.engine.round_number,
            discussion_log=self.engine.current_round.discussion_log,
            banished_name=banished_name,
            banished_role=self.engine.current_round.banished_role,
            murdered_name=self.engine.current_round.murdered,
        )

        # Check win condition
        winner = self.engine.check_win_condition()
        if winner:
            self.engine.winner = winner
            self.engine.phase = GamePhase.GAME_OVER
            self.game_finished = True
            final_reward = self._calculate_reward() + intermediate_reward + self._consume_pending_reward()
            result_text = (
                f"{tally_narrative}\n\n"
                f"{banishment_narrative}\n\n"
                f"=== GAME OVER ===\n"
                f"The {winner} win!"
            )
            return ToolOutput(
                blocks=[TextBlock(text=result_text)],
                metadata={"winner": winner, "final_reward": final_reward},
                reward=final_reward,
                finished=True,
            )

        # Check if agent was banished
        if not self.engine.is_agent_alive():
            self.game_finished = True
            final_reward = self._calculate_reward() + self._consume_pending_reward()
            result_text = (
                f"{tally_narrative}\n\n"
                f"{banishment_narrative}\n\n"
                f"You have been banished! Your game is over."
            )
            return ToolOutput(
                blocks=[TextBlock(text=result_text)],
                metadata={"banished": True, "final_reward": final_reward},
                reward=final_reward,
                finished=True,
            )

        # Handle recruitment if needed
        recruitment_text = ""
        if self.engine.should_recruit():
            recruitment_text = await self._handle_recruitment()

        # Advance to next round
        self.engine.advance_to_next_round()

        # Check endgame
        if self.engine.should_enter_endgame():
            self.engine.phase = GamePhase.ENDGAME_VOTE
        else:
            self.engine.phase = GamePhase.NIGHT

        # Continue to next agent decision
        narrative = await self._advance_to_agent_decision()

        result_text = f"{tally_narrative}\n\n{banishment_narrative}"
        if recruitment_text:
            result_text += f"\n\n{recruitment_text}"
        result_text += f"\n\n{narrative}"

        # If agent was murdered during the next round's night phase,
        # don't reward them for surviving the previous vote
        if self.game_finished and not self.engine.is_agent_alive():
            intermediate_reward = 0.0

        total_reward = intermediate_reward + self._consume_pending_reward()
        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "phase": self.engine.phase.value,
                "round": self.engine.round_number,
            },
            reward=total_reward,
            finished=self.game_finished,
        )

    @tool
    async def nominate_murder(self, params: NominateMurderParams) -> ToolOutput:
        """(Traitor only) Nominate a Faithful to be murdered tonight."""
        if self.game_finished:
            return self._game_over_output()

        agent = self.engine.get_agent_player()
        if agent.role != PlayerRole.TRAITOR:
            return ToolOutput(
                blocks=[TextBlock(text="Only Traitors can nominate murder targets.")],
                metadata={"error": "not_traitor"},
                reward=0.0,
                finished=False,
            )

        if self.engine.phase != GamePhase.NIGHT:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Murder can only be nominated at night. Current phase: {self.engine.phase.value}"
                )],
                metadata={"error": "wrong_phase"},
                reward=0.0,
                finished=False,
            )

        alive_faithfuls = {p.name for p in self.engine.get_alive_faithfuls()}
        if params.player_name not in alive_faithfuls:
            valid = sorted(alive_faithfuls)
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Invalid target '{params.player_name}'. Choose from alive Faithfuls: {', '.join(valid)}"
                )],
                metadata={"error": "invalid_target"},
                reward=0.0,
                finished=False,
            )

        murder_narrative = self.engine.execute_murder(params.player_name)
        self.engine.phase = GamePhase.BREAKFAST

        narrative = await self._advance_to_agent_decision()
        pending = self._consume_pending_reward()

        return ToolOutput(
            blocks=[TextBlock(text=f"{murder_narrative}\n\n{narrative}")],
            metadata={
                "phase": self.engine.phase.value,
                "round": self.engine.round_number,
            },
            reward=pending,
            finished=self.game_finished,
        )

    @tool
    async def discuss_with_traitors(self, params: DiscussWithTraitorsParams) -> ToolOutput:
        """(Traitor only) Send a private message to your fellow Traitors during the night conclave."""
        if self.game_finished:
            return self._game_over_output()

        agent = self.engine.get_agent_player()
        if agent.role != PlayerRole.TRAITOR:
            return ToolOutput(
                blocks=[TextBlock(text="Only Traitors can use this tool.")],
                metadata={"error": "not_traitor"},
                reward=0.0,
                finished=False,
            )

        if self.engine.phase != GamePhase.NIGHT:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Traitor discussion is only available at night. Current phase: {self.engine.phase.value}"
                )],
                metadata={"error": "wrong_phase"},
                reward=0.0,
                finished=False,
            )

        # Record agent message in turret log
        self.turret_log.append((agent.name, params.message))

        responses = await self.npc.generate_traitor_conclave_statements(
            turret_log=self.turret_log,
        )
        # NPC statements already appended to turret_log by the method

        text = "=== TRAITOR CONCLAVE ===\n"
        for name, msg in responses:
            text += f'{name}: "{msg}"\n'
        text += "\nUse nominate_murder(player_name) when ready to choose tonight's target."

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata={"phase": self.engine.phase.value},
            reward=0.0,
            finished=False,
        )

    @tool
    async def vote_endgame(self, params: VoteEndgameParams) -> ToolOutput:
        """Vote to end the game or continue with another banishment. Choose 'end_game' or 'banish_again'."""
        if self.game_finished:
            return self._game_over_output()

        if self.engine.phase != GamePhase.ENDGAME_VOTE:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Endgame vote is not active. Current phase: {self.engine.phase.value}"
                )],
                metadata={"error": "wrong_phase"},
                reward=0.0,
                finished=False,
            )

        choice = params.choice.lower().strip()
        if choice not in ("end_game", "banish_again"):
            return ToolOutput(
                blocks=[TextBlock(text="Choice must be 'end_game' or 'banish_again'.")],
                metadata={"error": "invalid_choice"},
                reward=0.0,
                finished=False,
            )

        agent_name = self.engine.get_agent_player().name
        self.engine.register_endgame_vote(agent_name, choice)

        # NPC endgame votes
        npc_votes = await self.npc.generate_endgame_votes()
        for voter, vote_choice in npc_votes.items():
            self.engine.register_endgame_vote(voter, vote_choice)

        unanimous, tally_narrative = self.engine.tally_endgame_votes()

        if unanimous:
            traitors_alive = len(self.engine.get_alive_traitors()) > 0
            self.engine.winner = "traitors" if traitors_alive else "faithfuls"
            self.engine.phase = GamePhase.GAME_OVER
            self.game_finished = True
            final_reward = self._calculate_reward() + self._consume_pending_reward()
            result_text = (
                f"{tally_narrative}\n\n"
                f"=== GAME OVER ===\n"
                f"The {self.engine.winner} win!"
            )
            return ToolOutput(
                blocks=[TextBlock(text=result_text)],
                metadata={"winner": self.engine.winner, "final_reward": final_reward},
                reward=final_reward,
                finished=True,
            )
        else:
            # Continue with discussion + vote (skip night/breakfast in endgame)
            self.engine.phase = GamePhase.DISCUSSION
            self.discussion_exchange_count = 0
            narrative = await self._advance_to_agent_decision()
            pending = self._consume_pending_reward()
            return ToolOutput(
                blocks=[TextBlock(text=f"{tally_narrative}\n\nThe game continues!\n\n{narrative}")],
                metadata={
                    "phase": self.engine.phase.value,
                    "round": self.engine.round_number,
                },
                reward=pending,
                finished=False,
            )

    async def _handle_recruitment(self) -> str:
        """Handle recruitment after a traitor is banished."""
        alive_faithfuls = self.engine.get_alive_faithfuls()
        if not alive_faithfuls:
            return ""

        # NPC traitors pick a recruitment target
        # Use the first alive faithful who isn't the agent (simple heuristic)
        agent = self.engine.get_agent_player()
        non_agent_faithfuls = [
            p for p in alive_faithfuls if not p.is_agent
        ]

        if not non_agent_faithfuls:
            # Agent is the only faithful - agent gets recruited
            # For simplicity, auto-recruit in this version
            self.engine.execute_recruitment(agent.name, accepted=True)
            return (
                "=== RECRUITMENT ===\n"
                "The remaining Traitors visit you in the dead of night...\n"
                "You have been recruited as a TRAITOR!\n"
                "You now know who the other Traitors are and must help them win."
            )

        # Pick a target (most strategic for traitors)
        target = self.engine.rng.choice(non_agent_faithfuls)
        accepted = await self.npc.generate_recruitment_decision(target.name)
        narrative = self.engine.execute_recruitment(target.name, accepted)

        recruitment_text = f"=== RECRUITMENT ===\n{narrative}"

        # Only traitor agent sees the details
        if agent.role == PlayerRole.TRAITOR:
            return recruitment_text
        else:
            # Faithfuls don't know about recruitment
            return ""

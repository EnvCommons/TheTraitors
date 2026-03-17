"""
Pure game state machine for The Traitors.
No LLM dependency - deterministic given seed + sequence of decisions.
"""

import random
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


class GamePhase(Enum):
    NIGHT = "night"
    BREAKFAST = "breakfast"
    MISSION = "mission"
    DISCUSSION = "discussion"
    ROUNDTABLE_VOTE = "roundtable_vote"
    RECRUITMENT = "recruitment"
    ENDGAME_VOTE = "endgame_vote"
    GAME_OVER = "game_over"


class PlayerStatus(Enum):
    ALIVE = "alive"
    MURDERED = "murdered"
    BANISHED = "banished"


class PlayerRole(Enum):
    FAITHFUL = "faithful"
    TRAITOR = "traitor"


@dataclass
class Player:
    index: int
    name: str
    role: PlayerRole
    status: PlayerStatus = PlayerStatus.ALIVE
    is_agent: bool = False


@dataclass
class RoundRecord:
    round_number: int
    murdered: Optional[str] = None
    banished: Optional[str] = None
    banished_role: Optional[str] = None
    votes: dict = field(default_factory=dict)
    discussion_log: list = field(default_factory=list)
    mission_reward: float = 0.0
    recruited: Optional[str] = None


class TraitorsGameEngine:
    NUM_PLAYERS = 22
    NUM_TRAITORS = 3
    MISSION_BASE_REWARD = 10000.0
    ENDGAME_THRESHOLD = 5
    MAX_ROUNDS = 15

    def __init__(self, seed: int, agent_role: str, agent_player_index: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.agent_role = PlayerRole(agent_role)
        self.agent_player_index = agent_player_index

        self.players: list[Player] = []
        self.phase: GamePhase = GamePhase.NIGHT
        self.round_number: int = 1
        self.prize_pot: float = 0.0
        self.round_history: list[RoundRecord] = []
        self.current_round: RoundRecord = RoundRecord(round_number=1)
        self.game_over: bool = False
        self.winner: Optional[str] = None

        # Phase-specific state
        self.current_votes: dict[str, str] = {}
        self.discussion_exchanges: int = 0
        self.max_discussion_exchanges: int = 3
        self.murder_target: Optional[str] = None
        self.endgame_votes: dict[str, str] = {}
        self.pending_recruitment: bool = False
        self.skip_night: bool = False  # True after recruitment (no murder that night)

    def setup_players(self, names: list[str], roles: list[str]) -> None:
        """Initialize players with given names and roles."""
        assert len(names) == self.NUM_PLAYERS
        assert len(roles) == self.NUM_PLAYERS

        self.players = []
        for i in range(self.NUM_PLAYERS):
            player = Player(
                index=i,
                name=names[i],
                role=PlayerRole(roles[i]),
                is_agent=(i == self.agent_player_index),
            )
            self.players.append(player)

    def get_alive_players(self) -> list[Player]:
        return [p for p in self.players if p.status == PlayerStatus.ALIVE]

    def get_alive_traitors(self) -> list[Player]:
        return [p for p in self.players
                if p.status == PlayerStatus.ALIVE and p.role == PlayerRole.TRAITOR]

    def get_alive_faithfuls(self) -> list[Player]:
        return [p for p in self.players
                if p.status == PlayerStatus.ALIVE and p.role == PlayerRole.FAITHFUL]

    def get_agent_player(self) -> Player:
        return self.players[self.agent_player_index]

    def is_agent_alive(self) -> bool:
        return self.get_agent_player().status == PlayerStatus.ALIVE

    def get_player_by_name(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None

    # --- Phase execution ---

    def execute_murder(self, target_name: str) -> str:
        """Traitors murder a player. Returns narrative text."""
        target = self.get_player_by_name(target_name)
        if target is None:
            return f"Error: Player '{target_name}' not found."
        if target.status != PlayerStatus.ALIVE:
            return f"Error: {target_name} is not alive."
        if target.role == PlayerRole.TRAITOR:
            return f"Error: Traitors cannot murder fellow Traitors."

        target.status = PlayerStatus.MURDERED
        self.murder_target = target_name
        self.current_round.murdered = target_name
        return f"The Traitors have chosen their victim: {target_name}."

    def execute_breakfast(self) -> str:
        """Reveal who was murdered. Returns narrative."""
        if self.murder_target:
            name = self.murder_target
            self.murder_target = None
            return (
                f"The contestants gather for breakfast at Ardross Castle. "
                f"One seat remains empty...\n\n"
                f"{name} has been murdered by the Traitors."
            )
        else:
            return (
                "The contestants gather for breakfast. "
                "Everyone is present -- no one was murdered last night!"
            )

    def execute_mission(self) -> tuple[float, str]:
        """Simulate a mission. Returns (reward_amount, narrative)."""
        # Randomised mission success (60-100% of base reward)
        success_rate = 0.6 + self.rng.random() * 0.4
        reward = round(self.MISSION_BASE_REWARD * success_rate, 2)
        self.prize_pot += reward
        self.current_round.mission_reward = reward

        mission_names = [
            "the Bridge of Betrayal", "the Maze of Mirrors",
            "the Tower Challenge", "the Castle Quest",
            "the Highland Trek", "the Vault of Secrets",
            "the Loch Challenge", "the Dungeon Escape",
            "the Garden of Lies", "the Banquet Riddle",
            "the Clock Tower Puzzle", "the Shield Retrieval",
            "the Mountain Pass", "the River Crossing",
            "the Crypt Cipher",
        ]
        mission_name = self.rng.choice(mission_names)

        narrative = (
            f"=== MISSION: {mission_name} ===\n"
            f"The contestants work together on {mission_name}.\n"
            f"They earn {reward:,.0f} for the prize pot.\n"
            f"Prize pot total: {self.prize_pot:,.0f}"
        )
        return reward, narrative

    def register_discussion_message(self, speaker_name: str, message: str) -> None:
        """Record a discussion statement."""
        self.current_round.discussion_log.append((speaker_name, message))

    def register_vote(self, voter_name: str, target_name: str) -> None:
        """Register a banishment vote."""
        self.current_votes[voter_name] = target_name
        self.current_round.votes[voter_name] = target_name

    def tally_votes(self) -> tuple[str, str]:
        """Tally votes. Returns (banished_name, narrative). Ties broken by RNG."""
        vote_counts: dict[str, int] = {}
        for voter, target in self.current_votes.items():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # Find maximum votes
        max_votes = max(vote_counts.values())
        candidates = [name for name, count in vote_counts.items() if count == max_votes]

        if len(candidates) == 1:
            banished_name = candidates[0]
        else:
            # Tie-break: random choice using seeded RNG
            banished_name = self.rng.choice(sorted(candidates))

        # Build narrative
        narrative_lines = ["=== VOTE RESULTS ==="]
        # Sort by vote count descending
        sorted_targets = sorted(vote_counts.items(), key=lambda x: -x[1])
        for target, count in sorted_targets:
            voters = [v for v, t in self.current_votes.items() if t == target]
            narrative_lines.append(
                f"  {target}: {count} vote{'s' if count != 1 else ''} "
                f"(from {', '.join(sorted(voters))})"
            )

        if len(candidates) > 1:
            narrative_lines.append(
                f"\nTie between {', '.join(sorted(candidates))}! "
                f"Resolved by chance."
            )

        narrative_lines.append(f"\n{banished_name} has been banished.")
        narrative = "\n".join(narrative_lines)

        # Clear votes for next round
        self.current_votes = {}

        return banished_name, narrative

    def execute_banishment(self, banished_name: str) -> str:
        """Banish a player. Returns narrative revealing their role."""
        player = self.get_player_by_name(banished_name)
        if player is None:
            return f"Error: Player '{banished_name}' not found."

        player.status = PlayerStatus.BANISHED
        self.current_round.banished = banished_name
        self.current_round.banished_role = player.role.value

        role_text = "a TRAITOR" if player.role == PlayerRole.TRAITOR else "a FAITHFUL"
        return (
            f"{banished_name} stands and walks to the door. "
            f"They turn to face the group...\n\n"
            f'"{banished_name} was {role_text}."\n\n'
            f"{'The group celebrates!' if player.role == PlayerRole.TRAITOR else 'A gasp fills the room. An innocent has been lost.'}"
        )

    def check_win_condition(self) -> Optional[str]:
        """Check if game is over. Returns 'faithfuls', 'traitors', or None."""
        alive_traitors = len(self.get_alive_traitors())
        alive_faithfuls = len(self.get_alive_faithfuls())

        # All traitors banished -> faithfuls win
        if alive_traitors == 0:
            return "faithfuls"

        # Traitors >= faithfuls -> traitors win (numerical parity/majority)
        if alive_traitors >= alive_faithfuls:
            return "traitors"

        # Max rounds exceeded -> check who remains
        if self.round_number > self.MAX_ROUNDS:
            if alive_traitors > 0:
                return "traitors"
            return "faithfuls"

        return None

    def should_enter_endgame(self) -> bool:
        """Check if we should offer endgame vote."""
        return len(self.get_alive_players()) <= self.ENDGAME_THRESHOLD

    def register_endgame_vote(self, voter_name: str, choice: str) -> None:
        """Register an endgame vote."""
        self.endgame_votes[voter_name] = choice

    def tally_endgame_votes(self) -> tuple[bool, str]:
        """Returns (end_game_unanimously, narrative)."""
        end_count = sum(1 for v in self.endgame_votes.values() if v == "end_game")
        banish_count = sum(1 for v in self.endgame_votes.values() if v == "banish_again")
        total = len(self.endgame_votes)

        narrative_lines = [
            "=== ENDGAME VOTE ===",
            f"End the game: {end_count} vote{'s' if end_count != 1 else ''}",
            f"Banish again: {banish_count} vote{'s' if banish_count != 1 else ''}",
        ]

        unanimous = (end_count == total and total > 0)
        if unanimous:
            narrative_lines.append("\nUnanimous! The game ends here.")
        else:
            narrative_lines.append(
                "\nNot unanimous. The game continues with another banishment round."
            )

        self.endgame_votes = {}
        return unanimous, "\n".join(narrative_lines)

    def should_recruit(self) -> bool:
        """After banishment, check if traitors should recruit."""
        alive_traitors = self.get_alive_traitors()
        alive_faithfuls = self.get_alive_faithfuls()
        # Recruit when traitor count drops to 1 or 2 and there are faithfuls available
        # and we haven't already entered endgame territory
        return (
            len(alive_traitors) <= 2
            and len(alive_traitors) > 0
            and len(alive_faithfuls) > 0
            and self.current_round.banished_role == "traitor"
            and not self.should_enter_endgame()
        )

    def execute_recruitment(self, target_name: str, accepted: bool) -> str:
        """Recruit a faithful as a traitor (or handle decline).
        Returns narrative."""
        player = self.get_player_by_name(target_name)
        if player is None:
            return f"Error: Player '{target_name}' not found."

        if accepted:
            player.role = PlayerRole.TRAITOR
            self.current_round.recruited = target_name
            self.skip_night = True  # No murder on recruitment night
            return (
                f"The Traitors visit {target_name} in the dead of night.\n"
                f"After a tense moment... {target_name} accepts.\n"
                f"{target_name} is now a TRAITOR."
            )
        else:
            self.skip_night = True  # No murder on recruitment night even if declined
            return (
                f"The Traitors visit {target_name} in the dead of night.\n"
                f"{target_name} declines the offer.\n"
                f"No murder takes place tonight."
            )

    def advance_to_next_round(self) -> None:
        """Archive current round, start new round."""
        self.round_history.append(self.current_round)
        self.round_number += 1
        self.current_round = RoundRecord(round_number=self.round_number)
        self.discussion_exchanges = 0

    def get_game_state_summary(self, for_player: Player) -> str:
        """Return game state visible to a specific player."""
        lines = [
            f"=== GAME STATE (Round {self.round_number}) ===",
            f"Phase: {self.phase.value}",
            f"Prize pot: {self.prize_pot:,.0f}",
            "",
            "ALIVE PLAYERS:",
        ]

        alive = self.get_alive_players()
        for p in alive:
            marker = " (you)" if p.is_agent else ""
            lines.append(f"  - {p.name}{marker}")

        lines.append(f"\nTotal alive: {len(alive)}")

        # Traitor-specific info
        if for_player.role == PlayerRole.TRAITOR:
            fellow_traitors = [
                p.name for p in self.get_alive_traitors()
                if p.name != for_player.name
            ]
            if fellow_traitors:
                lines.append(f"\nYour fellow Traitors: {', '.join(fellow_traitors)}")
            else:
                lines.append("\nYou are the sole remaining Traitor.")

        # Elimination history
        if self.round_history:
            lines.append("\nELIMINATION HISTORY:")
            for rr in self.round_history:
                if rr.murdered:
                    lines.append(f"  Round {rr.round_number}: {rr.murdered} was MURDERED")
                if rr.banished:
                    lines.append(
                        f"  Round {rr.round_number}: {rr.banished} was BANISHED "
                        f"(revealed as {rr.banished_role})"
                    )
                if rr.recruited:
                    if for_player.role == PlayerRole.TRAITOR:
                        lines.append(
                            f"  Round {rr.round_number}: {rr.recruited} was RECRUITED as a Traitor"
                        )

        # Voting history
        if self.round_history:
            lines.append("\nVOTING HISTORY:")
            for rr in self.round_history:
                if rr.votes:
                    lines.append(f"  Round {rr.round_number}:")
                    vote_summary = {}
                    for voter, target in rr.votes.items():
                        if target not in vote_summary:
                            vote_summary[target] = []
                        vote_summary[target].append(voter)
                    for target, voters in sorted(vote_summary.items(),
                                                  key=lambda x: -len(x[1])):
                        lines.append(
                            f"    {target}: {len(voters)} votes "
                            f"({', '.join(sorted(voters))})"
                        )

        return "\n".join(lines)

    def get_full_history_text(self) -> str:
        """Return full public game history for prompting.
        Includes both archived rounds and current round events.
        """
        all_rounds = list(self.round_history)
        # Include current round if it has any events recorded
        if (
            self.current_round
            and (self.current_round.murdered or self.current_round.banished
                 or self.current_round.mission_reward > 0)
        ):
            all_rounds.append(self.current_round)

        if not all_rounds:
            return "No rounds have been played yet."

        lines = []
        for rr in all_rounds:
            lines.append(f"--- Round {rr.round_number} ---")
            if rr.murdered:
                lines.append(f"  Murdered: {rr.murdered}")
            else:
                lines.append("  No murder (recruitment night or failed)")
            if rr.mission_reward > 0:
                lines.append(f"  Mission reward: {rr.mission_reward:,.0f}")
            if rr.banished:
                lines.append(
                    f"  Banished: {rr.banished} (was {rr.banished_role})"
                )
            if rr.votes:
                vote_counts: dict[str, int] = {}
                for target in rr.votes.values():
                    vote_counts[target] = vote_counts.get(target, 0) + 1
                top_votes = sorted(vote_counts.items(), key=lambda x: -x[1])[:3]
                vote_strs = [f"{name}({count})" for name, count in top_votes]
                lines.append(f"  Top votes: {', '.join(vote_strs)}")

        return "\n".join(lines)

    def get_discussion_log_text(self) -> str:
        """Return current round's discussion as text."""
        if not self.current_round.discussion_log:
            return "No discussion yet this round."

        lines = []
        for speaker, message in self.current_round.discussion_log:
            lines.append(f'{speaker}: "{message}"')
        return "\n".join(lines)

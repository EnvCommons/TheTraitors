"""
Comprehensive test suite for The Traitors environment.
Run with: pytest golden_tests.py -v
"""

import pytest
import random
from unittest.mock import AsyncMock, MagicMock, patch

from names import generate_player_setup
from game_engine import (
    TraitorsGameEngine, GamePhase, PlayerStatus, PlayerRole, Player, RoundRecord,
)


# =============================================================================
# Engine Unit Tests (no LLM dependency)
# =============================================================================


class TestEngineSetup:
    """Test 1: Player setup correctness."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_engine_player_count(self):
        engine = self._make_engine()
        assert len(engine.players) == 22

    def test_engine_traitor_count(self):
        engine = self._make_engine()
        traitors = engine.get_alive_traitors()
        assert len(traitors) == 3

    def test_engine_faithful_count(self):
        engine = self._make_engine()
        faithfuls = engine.get_alive_faithfuls()
        assert len(faithfuls) == 19

    def test_engine_agent_role_faithful(self):
        engine = self._make_engine(agent_role="faithful", agent_idx=0)
        agent = engine.get_agent_player()
        assert agent.role == PlayerRole.FAITHFUL
        assert agent.is_agent is True

    def test_engine_agent_role_traitor(self):
        engine = self._make_engine(agent_role="traitor", agent_idx=5)
        agent = engine.get_agent_player()
        assert agent.role == PlayerRole.TRAITOR
        assert agent.is_agent is True

    def test_engine_deterministic(self):
        engine1 = self._make_engine(seed=42)
        engine2 = self._make_engine(seed=42)
        names1 = [p.name for p in engine1.players]
        names2 = [p.name for p in engine2.players]
        assert names1 == names2

    def test_engine_different_seeds(self):
        engine1 = self._make_engine(seed=42)
        engine2 = self._make_engine(seed=99)
        names1 = [p.name for p in engine1.players]
        names2 = [p.name for p in engine2.players]
        assert names1 != names2

    def test_engine_all_names_unique(self):
        engine = self._make_engine()
        names = [p.name for p in engine.players]
        assert len(set(names)) == 22

    def test_engine_initial_phase(self):
        engine = self._make_engine()
        assert engine.phase == GamePhase.NIGHT

    def test_engine_initial_prize_pot(self):
        engine = self._make_engine()
        assert engine.prize_pot == 0.0

    def test_engine_initial_round(self):
        engine = self._make_engine()
        assert engine.round_number == 1


class TestEngineMurder:
    """Test 2: Murder mechanics."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_murder_removes_player(self):
        engine = self._make_engine()
        faithful = engine.get_alive_faithfuls()[0]
        engine.execute_murder(faithful.name)
        assert faithful.status == PlayerStatus.MURDERED

    def test_murder_reduces_alive_count(self):
        engine = self._make_engine()
        initial_count = len(engine.get_alive_players())
        faithful = engine.get_alive_faithfuls()[0]
        engine.execute_murder(faithful.name)
        assert len(engine.get_alive_players()) == initial_count - 1

    def test_murder_only_faithfuls(self):
        engine = self._make_engine()
        traitor = engine.get_alive_traitors()[0]
        result = engine.execute_murder(traitor.name)
        assert "cannot murder" in result.lower()
        assert traitor.status == PlayerStatus.ALIVE

    def test_murder_records_in_round(self):
        engine = self._make_engine()
        faithful = engine.get_alive_faithfuls()[0]
        engine.execute_murder(faithful.name)
        assert engine.current_round.murdered == faithful.name

    def test_murder_nonexistent_player(self):
        engine = self._make_engine()
        result = engine.execute_murder("NonexistentPerson")
        assert "not found" in result.lower()

    def test_breakfast_reveals_murder(self):
        engine = self._make_engine()
        victim = engine.get_alive_faithfuls()[0]
        engine.execute_murder(victim.name)
        narrative = engine.execute_breakfast()
        assert victim.name in narrative
        assert "murdered" in narrative.lower()

    def test_breakfast_no_murder(self):
        engine = self._make_engine()
        narrative = engine.execute_breakfast()
        assert "everyone is present" in narrative.lower()


class TestEngineVoting:
    """Test 3: Vote tallying."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_vote_majority(self):
        engine = self._make_engine()
        alive = engine.get_alive_players()
        target = alive[1].name
        other = alive[2].name
        # Give target a clear majority
        for i, p in enumerate(alive):
            if i < 15:
                engine.register_vote(p.name, target)
            else:
                engine.register_vote(p.name, other)

        banished, narrative = engine.tally_votes()
        assert banished == target
        assert target in narrative

    def test_vote_tie_breaking(self):
        """With deterministic RNG, ties should be resolved consistently."""
        engine = self._make_engine(seed=42)
        alive = engine.get_alive_players()
        target_a = alive[1].name
        target_b = alive[2].name
        # Split votes evenly
        for i, p in enumerate(alive):
            if i % 2 == 0:
                engine.register_vote(p.name, target_a)
            else:
                engine.register_vote(p.name, target_b)

        banished, _ = engine.tally_votes()
        assert banished in (target_a, target_b)

        # Running again with same seed should give same result
        engine2 = self._make_engine(seed=42)
        alive2 = engine2.get_alive_players()
        target_a2 = alive2[1].name
        target_b2 = alive2[2].name
        for i, p in enumerate(alive2):
            if i % 2 == 0:
                engine2.register_vote(p.name, target_a2)
            else:
                engine2.register_vote(p.name, target_b2)

        banished2, _ = engine2.tally_votes()
        assert banished == banished2

    def test_vote_clears_after_tally(self):
        engine = self._make_engine()
        alive = engine.get_alive_players()
        engine.register_vote(alive[0].name, alive[1].name)
        engine.tally_votes()
        assert len(engine.current_votes) == 0

    def test_vote_records_in_round(self):
        engine = self._make_engine()
        alive = engine.get_alive_players()
        engine.register_vote(alive[0].name, alive[1].name)
        assert alive[0].name in engine.current_round.votes

    def test_vote_many_voters(self):
        """Test with all 22 players voting."""
        engine = self._make_engine()
        alive = engine.get_alive_players()
        target = alive[0].name
        for p in alive[1:]:  # 21 votes for 1 target
            engine.register_vote(p.name, target)
        engine.register_vote(alive[0].name, alive[1].name)  # 1 vote for another

        banished, _ = engine.tally_votes()
        assert banished == target


class TestEngineBanishment:
    """Test 4: Banishment mechanics."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_banishment_updates_status(self):
        engine = self._make_engine()
        target = engine.get_alive_faithfuls()[0]
        engine.execute_banishment(target.name)
        assert target.status == PlayerStatus.BANISHED

    def test_banishment_reveals_role_faithful(self):
        engine = self._make_engine()
        faithful = engine.get_alive_faithfuls()[0]
        narrative = engine.execute_banishment(faithful.name)
        assert "FAITHFUL" in narrative
        assert engine.current_round.banished_role == "faithful"

    def test_banishment_reveals_role_traitor(self):
        engine = self._make_engine()
        traitor = engine.get_alive_traitors()[0]
        narrative = engine.execute_banishment(traitor.name)
        assert "TRAITOR" in narrative
        assert engine.current_round.banished_role == "traitor"

    def test_banishment_records_in_round(self):
        engine = self._make_engine()
        target = engine.get_alive_players()[1]
        engine.execute_banishment(target.name)
        assert engine.current_round.banished == target.name


class TestEngineWinConditions:
    """Test 5: Win conditions."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_faithfuls_win_all_traitors_banished(self):
        engine = self._make_engine()
        for traitor in engine.get_alive_traitors():
            traitor.status = PlayerStatus.BANISHED
        result = engine.check_win_condition()
        assert result == "faithfuls"

    def test_traitors_win_parity(self):
        """Traitors win when traitors >= faithfuls."""
        engine = self._make_engine()
        # Kill all but 3 faithfuls
        faithfuls = engine.get_alive_faithfuls()
        for f in faithfuls[:16]:  # Kill 16, leaving 3
            f.status = PlayerStatus.MURDERED
        # 3 traitors vs 3 faithfuls => traitors win
        result = engine.check_win_condition()
        assert result == "traitors"

    def test_traitors_win_majority(self):
        engine = self._make_engine()
        faithfuls = engine.get_alive_faithfuls()
        for f in faithfuls[:17]:  # Kill 17, leaving 2
            f.status = PlayerStatus.MURDERED
        # 3 traitors vs 2 faithfuls
        result = engine.check_win_condition()
        assert result == "traitors"

    def test_no_winner_yet(self):
        engine = self._make_engine()
        result = engine.check_win_condition()
        assert result is None

    def test_max_rounds_triggers_end(self):
        engine = self._make_engine()
        engine.round_number = engine.MAX_ROUNDS + 1
        result = engine.check_win_condition()
        assert result == "traitors"  # Traitors still alive

    def test_max_rounds_faithfuls_win(self):
        engine = self._make_engine()
        engine.round_number = engine.MAX_ROUNDS + 1
        for t in engine.get_alive_traitors():
            t.status = PlayerStatus.BANISHED
        result = engine.check_win_condition()
        assert result == "faithfuls"


class TestEngineEndgame:
    """Test 6: Endgame mechanics."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_endgame_triggers_at_threshold(self):
        engine = self._make_engine()
        # Kill all but 5 players
        alive = engine.get_alive_players()
        for p in alive[5:]:
            p.status = PlayerStatus.MURDERED
        assert engine.should_enter_endgame() is True

    def test_endgame_not_triggered_above_threshold(self):
        engine = self._make_engine()
        alive = engine.get_alive_players()
        for p in alive[6:]:
            p.status = PlayerStatus.MURDERED
        assert engine.should_enter_endgame() is False

    def test_endgame_unanimous_end(self):
        engine = self._make_engine()
        alive = engine.get_alive_players()[:5]
        for p in engine.get_alive_players()[5:]:
            p.status = PlayerStatus.MURDERED
        for p in alive:
            engine.register_endgame_vote(p.name, "end_game")
        unanimous, narrative = engine.tally_endgame_votes()
        assert unanimous is True
        assert "unanimous" in narrative.lower()

    def test_endgame_not_unanimous(self):
        engine = self._make_engine()
        alive = engine.get_alive_players()[:5]
        for p in engine.get_alive_players()[5:]:
            p.status = PlayerStatus.MURDERED
        for i, p in enumerate(alive):
            if i == 0:
                engine.register_endgame_vote(p.name, "banish_again")
            else:
                engine.register_endgame_vote(p.name, "end_game")
        unanimous, narrative = engine.tally_endgame_votes()
        assert unanimous is False
        assert "not unanimous" in narrative.lower()

    def test_endgame_clears_votes(self):
        engine = self._make_engine()
        engine.register_endgame_vote("SomePlayer", "end_game")
        engine.tally_endgame_votes()
        assert len(engine.endgame_votes) == 0


class TestEngineRecruitment:
    """Test 7: Recruitment mechanics."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_recruitment_triggered_after_traitor_banishment(self):
        engine = self._make_engine()
        traitor = engine.get_alive_traitors()[0]
        engine.execute_banishment(traitor.name)
        assert engine.should_recruit() is True

    def test_recruitment_not_triggered_after_faithful_banishment(self):
        engine = self._make_engine()
        faithful = engine.get_alive_faithfuls()[0]
        engine.execute_banishment(faithful.name)
        assert engine.should_recruit() is False

    def test_recruitment_changes_role(self):
        engine = self._make_engine()
        target = engine.get_alive_faithfuls()[0]
        assert target.role == PlayerRole.FAITHFUL
        engine.execute_recruitment(target.name, accepted=True)
        assert target.role == PlayerRole.TRAITOR

    def test_recruitment_declined_keeps_role(self):
        engine = self._make_engine()
        target = engine.get_alive_faithfuls()[0]
        engine.execute_recruitment(target.name, accepted=False)
        assert target.role == PlayerRole.FAITHFUL

    def test_recruitment_sets_skip_night(self):
        engine = self._make_engine()
        target = engine.get_alive_faithfuls()[0]
        engine.execute_recruitment(target.name, accepted=True)
        assert engine.skip_night is True

    def test_recruitment_not_triggered_in_endgame(self):
        """Recruitment shouldn't trigger if we're at endgame threshold."""
        engine = self._make_engine()
        # Kill everyone except 4 (below threshold)
        alive = engine.get_alive_players()
        for p in alive[4:]:
            p.status = PlayerStatus.MURDERED
        # Now banish a traitor
        traitors = engine.get_alive_traitors()
        if traitors:
            engine.execute_banishment(traitors[0].name)
            # Should NOT recruit because <= 5 players remain
            # (after banishment we have 3 players left)
            assert engine.should_recruit() is False


class TestEnginePhaseTransitions:
    """Test 8: Phase transition correctness."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_initial_phase_is_night(self):
        engine = self._make_engine()
        assert engine.phase == GamePhase.NIGHT

    def test_advance_round(self):
        engine = self._make_engine()
        assert engine.round_number == 1
        engine.advance_to_next_round()
        assert engine.round_number == 2
        assert len(engine.round_history) == 1


class TestEngineReward:
    """Test 9: Reward calculation helpers (used by environment)."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_mission_adds_to_prize_pot(self):
        engine = self._make_engine()
        reward, narrative = engine.execute_mission()
        assert reward > 0
        assert engine.prize_pot == reward
        assert "prize pot" in narrative.lower()

    def test_mission_deterministic(self):
        engine1 = self._make_engine(seed=42)
        engine2 = self._make_engine(seed=42)
        r1, _ = engine1.execute_mission()
        r2, _ = engine2.execute_mission()
        assert r1 == r2


class TestEngineTaskGeneration:
    """Test 10: Task generation correctness."""

    def test_seeds_count(self):
        """Verify we can generate 100 tasks per split."""
        # This tests the planned list_tasks logic
        train_tasks = []
        for seed in range(100):
            role = "faithful" if seed < 50 else "traitor"
            agent_idx = seed % 22
            train_tasks.append({
                "id": f"train_{seed:03d}_{role}",
                "seed": seed,
                "agent_role": role,
                "agent_player_index": agent_idx,
            })
        assert len(train_tasks) == 100
        # 50 faithful, 50 traitor
        assert sum(1 for t in train_tasks if t["agent_role"] == "faithful") == 50
        assert sum(1 for t in train_tasks if t["agent_role"] == "traitor") == 50

    def test_test_split_seeds(self):
        test_tasks = []
        for seed in range(10000, 10100):
            role = "faithful" if (seed - 10000) < 50 else "traitor"
            agent_idx = (seed - 10000) % 22
            test_tasks.append({
                "id": f"test_{seed}_{role}",
                "seed": seed,
                "agent_role": role,
                "agent_player_index": agent_idx,
            })
        assert len(test_tasks) == 100
        # No overlap with train seeds
        train_seeds = set(range(100))
        test_seeds = set(range(10000, 10100))
        assert len(train_seeds & test_seeds) == 0


class TestEngine22PlayerEdgeCases:
    """Test 11: 22-player specific edge cases."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_many_rounds_elimination(self):
        """Simulate many rounds of elimination."""
        engine = self._make_engine()
        rounds = 0
        while len(engine.get_alive_players()) > 5 and rounds < 20:
            # Murder a faithful
            faithfuls = engine.get_alive_faithfuls()
            if faithfuls:
                engine.execute_murder(faithfuls[0].name)
            # Banish someone
            alive = engine.get_alive_players()
            if alive:
                engine.execute_banishment(alive[0].name)
            engine.advance_to_next_round()
            rounds += 1
        assert len(engine.get_alive_players()) <= 10

    def test_all_faithfuls_murdered_traitors_win(self):
        """Edge case: all faithfuls murdered before endgame."""
        engine = self._make_engine()
        for f in engine.get_alive_faithfuls():
            f.status = PlayerStatus.MURDERED
        assert engine.check_win_condition() == "traitors"

    def test_game_state_summary_large_player_count(self):
        engine = self._make_engine()
        agent = engine.get_agent_player()
        summary = engine.get_game_state_summary(agent)
        assert "22" not in summary or "Total alive: 22" in summary
        assert agent.name in summary


class TestEngineGameTermination:
    """Test 12: Game terminates within MAX_ROUNDS."""

    def _make_engine(self, seed=42, agent_role="faithful", agent_idx=0):
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        engine.setup_players(names, roles)
        return engine

    def test_max_rounds_forces_termination(self):
        engine = self._make_engine()
        engine.round_number = engine.MAX_ROUNDS + 1
        result = engine.check_win_condition()
        assert result is not None

    def test_game_history_text(self):
        engine = self._make_engine()
        # Empty history
        text = engine.get_full_history_text()
        assert "no rounds" in text.lower()

        # Add a round
        faithful = engine.get_alive_faithfuls()[0]
        engine.execute_murder(faithful.name)
        engine.register_vote("TestVoter", faithful.name)
        engine.current_round.mission_reward = 5000
        engine.advance_to_next_round()

        text = engine.get_full_history_text()
        assert "round 1" in text.lower()
        assert faithful.name.lower() in text.lower()

    def test_discussion_log(self):
        engine = self._make_engine()
        engine.register_discussion_message("Alice", "I think Bob is suspicious")
        text = engine.get_discussion_log_text()
        assert "Alice" in text
        assert "suspicious" in text

    def test_get_player_by_name(self):
        engine = self._make_engine()
        first_player = engine.players[0]
        found = engine.get_player_by_name(first_player.name)
        assert found is not None
        assert found.name == first_player.name

    def test_get_player_by_name_not_found(self):
        engine = self._make_engine()
        found = engine.get_player_by_name("NonexistentPerson")
        assert found is None


# =============================================================================
# NPC Unit Tests (with mocked LLM)
# =============================================================================


class TestNPCMemory:
    """NPC memory and suspicion tracking tests."""

    def _make_engine_and_npc(self, seed=42, agent_role="faithful", agent_idx=0):
        from npc import NPCController
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        personalities = {s[0]: s[1] for s in setup}
        engine.setup_players(names, roles)

        mock_client = AsyncMock()
        rng = random.Random(seed)
        npc = NPCController(mock_client, engine, rng)
        npc.initialize_memories(engine.players, personalities)

        return engine, npc, mock_client

    def test_npc_memories_initialized_for_all_npcs(self):
        engine, npc, _ = self._make_engine_and_npc()
        # 21 NPCs (all except agent)
        assert len(npc.memories) == 21

    def test_npc_memory_not_created_for_agent(self):
        engine, npc, _ = self._make_engine_and_npc()
        agent = engine.get_agent_player()
        assert agent.name not in npc.memories

    def test_npc_suspicion_scores_initialized(self):
        engine, npc, _ = self._make_engine_and_npc()
        for name, mem in npc.memories.items():
            # Should have suspicion scores for all other players
            assert len(mem.suspicion_scores) == 21  # 22 players - self
            # Initial suspicion should be low
            for score in mem.suspicion_scores.values():
                assert score == pytest.approx(0.2)

    def test_npc_trust_scores_initialized(self):
        engine, npc, _ = self._make_engine_and_npc()
        for name, mem in npc.memories.items():
            for score in mem.trust_scores.values():
                assert score == pytest.approx(0.5)

    def test_suspicion_update_after_traitor_banishment(self):
        engine, npc, _ = self._make_engine_and_npc()
        # Get a faithful NPC
        faithful_npc = None
        for name, mem in npc.memories.items():
            if mem.role == PlayerRole.FAITHFUL:
                faithful_npc = mem
                break
        assert faithful_npc is not None

        # Record initial suspicion
        other = list(faithful_npc.suspicion_scores.keys())[0]
        initial_suspicion = faithful_npc.suspicion_scores[other]

        # Update after traitor banishment
        npc.update_suspicions_from_round(
            discussion_log=[],
            banished_name="SomeTraitor",
            banished_role="traitor",
        )

        # Suspicion should decrease
        assert faithful_npc.suspicion_scores[other] < initial_suspicion

    def test_suspicion_update_after_faithful_banishment(self):
        engine, npc, _ = self._make_engine_and_npc()
        faithful_npc = None
        for name, mem in npc.memories.items():
            if mem.role == PlayerRole.FAITHFUL:
                faithful_npc = mem
                break

        other = list(faithful_npc.suspicion_scores.keys())[0]
        initial_suspicion = faithful_npc.suspicion_scores[other]

        npc.update_suspicions_from_round(
            discussion_log=[],
            banished_name="SomeFaithful",
            banished_role="faithful",
        )

        # Suspicion should increase (paranoia)
        assert faithful_npc.suspicion_scores[other] > initial_suspicion

    def test_traitor_npc_keeps_fellow_traitors_at_zero_suspicion(self):
        engine, npc, _ = self._make_engine_and_npc()
        traitor_npcs = [
            (name, mem) for name, mem in npc.memories.items()
            if mem.role == PlayerRole.TRAITOR
        ]
        assert len(traitor_npcs) >= 1  # At least some NPC traitors

        # Run suspicion update
        npc.update_suspicions_from_round(
            discussion_log=[("Someone", "I suspect everyone")],
            banished_name=None,
            banished_role=None,
        )

        # Check traitor NPC has 0 suspicion for other traitors
        for name, mem in traitor_npcs:
            for other_name, score in mem.suspicion_scores.items():
                other_player = engine.get_player_by_name(other_name)
                if other_player and other_player.role == PlayerRole.TRAITOR:
                    assert score == 0.0


class TestNPCSpeakerSelection:
    """Test speaker selection for discussions."""

    def _make_engine_and_npc(self, seed=42, agent_role="faithful", agent_idx=0):
        from npc import NPCController
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        personalities = {s[0]: s[1] for s in setup}
        engine.setup_players(names, roles)

        mock_client = AsyncMock()
        rng = random.Random(seed)
        npc = NPCController(mock_client, engine, rng)
        npc.initialize_memories(engine.players, personalities)
        return engine, npc, mock_client

    def test_speaker_selection_returns_correct_count(self):
        engine, npc, _ = self._make_engine_and_npc()
        speakers = npc._select_speakers(exchange_number=0, num_speakers=6)
        assert len(speakers) == 6

    def test_speaker_selection_excludes_agent(self):
        engine, npc, _ = self._make_engine_and_npc()
        speakers = npc._select_speakers(exchange_number=0, num_speakers=6)
        agent = engine.get_agent_player()
        speaker_names = [s.name for s in speakers]
        assert agent.name not in speaker_names

    def test_speaker_selection_all_alive(self):
        engine, npc, _ = self._make_engine_and_npc()
        speakers = npc._select_speakers(exchange_number=0)
        for s in speakers:
            assert s.status == PlayerStatus.ALIVE

    def test_speaker_selection_with_few_alive(self):
        engine, npc, _ = self._make_engine_and_npc()
        # Kill most players
        alive = engine.get_alive_players()
        for p in alive[3:]:
            p.status = PlayerStatus.MURDERED
        speakers = npc._select_speakers(exchange_number=0, num_speakers=6)
        # Should return all alive NPCs (less than 6)
        alive_npcs = [p for p in engine.get_alive_players() if not p.is_agent]
        assert len(speakers) == len(alive_npcs)


@pytest.mark.asyncio
class TestNPCLLMCalls:
    """Test NPC LLM-driven methods with mocked client."""

    def _make_engine_and_npc(self, seed=42, agent_role="faithful", agent_idx=0):
        from npc import NPCController
        engine = TraitorsGameEngine(seed=seed, agent_role=agent_role, agent_player_index=agent_idx)
        setup = generate_player_setup(seed, agent_idx, agent_role)
        names = [s[0] for s in setup]
        roles = [s[2] for s in setup]
        personalities = {s[0]: s[1] for s in setup}
        engine.setup_players(names, roles)

        mock_client = AsyncMock()
        rng = random.Random(seed)
        npc = NPCController(mock_client, engine, rng)
        npc.initialize_memories(engine.players, personalities)
        return engine, npc, mock_client

    def _setup_mock_response(self, mock_client, text):
        """Configure mock client to return a specific text."""
        mock_message = MagicMock()
        mock_message.content = text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    async def test_discussion_returns_statements(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        self._setup_mock_response(mock_client, "I think someone is being very suspicious today.")
        statements = await npc.generate_discussion_statements(
            round_number=1, exchange_number=0, prior_statements=[]
        )
        assert len(statements) > 0
        for name, msg in statements:
            assert isinstance(name, str)
            assert isinstance(msg, str)
            assert len(msg) > 0

    async def test_vote_generation_returns_valid_names(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        # Mock returns a valid player name
        faithful = engine.get_alive_faithfuls()[0]
        self._setup_mock_response(mock_client, faithful.name)
        votes = await npc.generate_npc_votes()
        assert len(votes) > 0
        alive_names = {p.name for p in engine.get_alive_players()}
        for voter, target in votes.items():
            assert target in alive_names
            assert voter != target

    async def test_vote_fallback_on_invalid_response(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        self._setup_mock_response(mock_client, "INVALID_NAME_GARBAGE")
        votes = await npc.generate_npc_votes()
        # Should still produce votes (via fallback)
        assert len(votes) > 0
        alive_names = {p.name for p in engine.get_alive_players()}
        for voter, target in votes.items():
            assert target in alive_names

    async def test_murder_selection_returns_faithful(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        faithful = engine.get_alive_faithfuls()[0]
        self._setup_mock_response(mock_client, faithful.name)
        target = await npc.generate_murder_selection()
        faithful_names = {p.name for p in engine.get_alive_faithfuls()}
        assert target in faithful_names

    async def test_murder_fallback_on_invalid_response(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        self._setup_mock_response(mock_client, "NOBODY")
        target = await npc.generate_murder_selection()
        faithful_names = {p.name for p in engine.get_alive_faithfuls()}
        assert target in faithful_names

    async def test_traitor_conclave_returns_statements(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        self._setup_mock_response(mock_client, "We should target the loud one.")
        statements = await npc.generate_traitor_conclave_statements()
        # Should have statements from NPC traitors
        npc_traitors = [p for p in engine.get_alive_traitors() if not p.is_agent]
        assert len(statements) == len(npc_traitors)

    async def test_endgame_votes_valid(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        self._setup_mock_response(mock_client, "end_game")
        votes = await npc.generate_endgame_votes()
        for name, vote in votes.items():
            assert vote in ("end_game", "banish_again")

    async def test_endgame_vote_fallback(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        self._setup_mock_response(mock_client, "INVALID")
        votes = await npc.generate_endgame_votes()
        for name, vote in votes.items():
            assert vote in ("end_game", "banish_again")

    async def test_recruitment_decision(self):
        engine, npc, mock_client = self._make_engine_and_npc()
        self._setup_mock_response(mock_client, "accept")
        faithful = engine.get_alive_faithfuls()[0]
        # Only test if the faithful is an NPC
        if faithful.name in npc.memories:
            result = await npc.generate_recruitment_decision(faithful.name)
            assert isinstance(result, bool)


# =============================================================================
# Integration Tests (mock LLM)
# =============================================================================


def _make_mock_env(seed=42, agent_role="faithful", agent_idx=0):
    """Create a TraitorsEnvironment with mocked OpenAI client."""
    from thetraitors import TraitorsEnvironment

    task_spec = {
        "id": f"test_{seed}_{agent_role}",
        "seed": seed,
        "agent_role": agent_role,
        "agent_player_index": agent_idx,
    }
    secrets = {"openai_api_key": "test-key-mock"}

    with patch("openai.AsyncClient") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        env = TraitorsEnvironment(task_spec, secrets)

    # Replace the client on the NPC controller too
    env.npc.client = mock_client

    return env, mock_client


def _setup_mock_llm(mock_client, response_text):
    """Configure mock to return given text for all LLM calls."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)


def _setup_mock_llm_safe(env, mock_client):
    """Configure mock to return a non-agent faithful name for murder,
    and generic discussion text for other calls.
    This prevents the agent from being murdered during get_prompt().
    """
    agent_name = env.engine.get_agent_player().name
    # Find a faithful that is NOT the agent
    safe_target = None
    for p in env.engine.get_alive_faithfuls():
        if p.name != agent_name:
            safe_target = p.name
            break
    if safe_target is None:
        safe_target = env.engine.get_alive_players()[1].name

    mock_message = MagicMock()
    mock_message.content = safe_target
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)


@pytest.mark.asyncio
class TestIntegrationFaithful:
    """Test 13-17: Full game integration as Faithful."""

    async def test_get_prompt_faithful(self):
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        # Mock NPCs to return discussion statements and murder target
        _setup_mock_llm_safe(env, mock_client)
        prompt_blocks = await env.get_prompt()
        assert len(prompt_blocks) == 1
        text = prompt_blocks[0].text
        assert "FAITHFUL" in text
        assert env.engine.get_agent_player().name in text

    async def test_speak_during_discussion(self):
        from thetraitors import SpeakParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        # Agent speaks
        if not env.game_finished:
            result = await env.speak(SpeakParams(message="I think someone is suspicious!"))
            assert len(result.blocks) > 0

    async def test_speak_wrong_phase(self):
        from thetraitors import SpeakParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        if not env.game_finished:
            env.engine.phase = GamePhase.ROUNDTABLE_VOTE
            result = await env.speak(SpeakParams(message="test"))
            assert "wrong_phase" in str(result.metadata.get("error", ""))

    async def test_view_game_state(self):
        from thetraitors import ViewGameStateParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        result = await env.view_game_state(ViewGameStateParams())
        # view_game_state should work even if game is over
        text = result.blocks[0].text
        assert "ALIVE PLAYERS" in text or "game" in text.lower()

    async def test_cast_vote_invalid_target(self):
        from thetraitors import CastVoteParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        if not env.game_finished:
            env.engine.phase = GamePhase.ROUNDTABLE_VOTE
            result = await env.cast_vote(CastVoteParams(player_name="NonexistentPlayer"))
            assert "invalid_target" in str(result.metadata.get("error", ""))

    async def test_cast_vote_cannot_vote_self(self):
        from thetraitors import CastVoteParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        if not env.game_finished:
            env.engine.phase = GamePhase.ROUNDTABLE_VOTE
            agent_name = env.engine.get_agent_player().name
            result = await env.cast_vote(CastVoteParams(player_name=agent_name))
            assert "invalid_target" in str(result.metadata.get("error", ""))


@pytest.mark.asyncio
class TestIntegrationTraitor:
    """Test 14: Full game integration as Traitor."""

    async def test_get_prompt_traitor(self):
        env, mock_client = _make_mock_env(seed=50, agent_role="traitor", agent_idx=5)
        _setup_mock_llm_safe(env, mock_client)
        prompt_blocks = await env.get_prompt()
        assert len(prompt_blocks) == 1
        text = prompt_blocks[0].text
        assert "TRAITOR" in text
        # Should see fellow traitor names
        assert "fellow Traitor" in text or "Turret" in text

    async def test_nominate_murder_valid(self):
        from thetraitors import NominateMurderParams
        env, mock_client = _make_mock_env(seed=50, agent_role="traitor", agent_idx=5)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        # Should be in NIGHT phase for traitor
        if env.engine.phase == GamePhase.NIGHT:
            target = env.engine.get_alive_faithfuls()[0].name
            result = await env.nominate_murder(NominateMurderParams(player_name=target))
            assert len(result.blocks) > 0

    async def test_nominate_murder_non_traitor(self):
        from thetraitors import NominateMurderParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        if not env.game_finished:
            env.engine.phase = GamePhase.NIGHT
            target = env.engine.get_alive_faithfuls()[0].name
            result = await env.nominate_murder(
                NominateMurderParams(player_name=target)
            )
            assert "not_traitor" in str(result.metadata.get("error", ""))

    async def test_discuss_with_traitors(self):
        from thetraitors import DiscussWithTraitorsParams
        env, mock_client = _make_mock_env(seed=50, agent_role="traitor", agent_idx=5)
        faithful = env.engine.get_alive_faithfuls()[0]
        _setup_mock_llm(mock_client, "Let's target the loud one.")
        await env.get_prompt()

        if env.engine.phase == GamePhase.NIGHT:
            result = await env.discuss_with_traitors(
                DiscussWithTraitorsParams(message="Who should we target?")
            )
            assert "CONCLAVE" in result.blocks[0].text

    async def test_discuss_with_traitors_non_traitor(self):
        from thetraitors import DiscussWithTraitorsParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        if not env.game_finished:
            result = await env.discuss_with_traitors(
                DiscussWithTraitorsParams(message="test")
            )
            assert "not_traitor" in str(result.metadata.get("error", ""))


@pytest.mark.asyncio
class TestIntegrationEndgame:
    """Test 20: Endgame flow."""

    async def test_vote_endgame_invalid_choice(self):
        from thetraitors import VoteEndgameParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        if not env.game_finished:
            env.engine.phase = GamePhase.ENDGAME_VOTE
            result = await env.vote_endgame(VoteEndgameParams(choice="maybe"))
            assert "invalid_choice" in str(result.metadata.get("error", ""))

    async def test_vote_endgame_wrong_phase(self):
        from thetraitors import VoteEndgameParams
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        if not env.game_finished:
            result = await env.vote_endgame(VoteEndgameParams(choice="end_game"))
            assert "wrong_phase" in str(result.metadata.get("error", ""))


@pytest.mark.asyncio
class TestIntegrationViewState:
    """Test 21: View game state role-appropriate info."""

    async def test_traitor_sees_fellow_traitors(self):
        from thetraitors import ViewGameStateParams
        env, mock_client = _make_mock_env(seed=50, agent_role="traitor", agent_idx=5)
        _setup_mock_llm_safe(env, mock_client)
        await env.get_prompt()

        result = await env.view_game_state(ViewGameStateParams())
        text = result.blocks[0].text
        # Traitor should see fellow traitor info
        assert "Traitor" in text


@pytest.mark.asyncio
class TestIntegrationReward:
    """Test reward calculation scenarios."""

    async def test_reward_in_range(self):
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        reward = env._calculate_reward()
        assert 0.0 <= reward <= 1.0

    async def test_reward_increases_with_survival(self):
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        r1 = env._calculate_reward()
        env.engine.round_number = 5
        r2 = env._calculate_reward()
        assert r2 >= r1

    async def test_reward_win_condition(self):
        env, mock_client = _make_mock_env(seed=42, agent_role="faithful", agent_idx=0)
        # Banish all traitors
        for t in env.engine.get_alive_traitors():
            t.status = PlayerStatus.BANISHED
        env.engine.winner = "faithfuls"
        env.engine.prize_pot = 50000.0
        env.engine.round_number = 5
        reward = env._calculate_reward()
        # Should be substantial (survival + win + pot)
        assert reward > 0.5


# =============================================================================
# Smoke Tests
# =============================================================================


class TestSmoke:
    """Test 24-28: Smoke tests for environment setup."""

    def test_list_splits(self):
        from thetraitors import TraitorsEnvironment
        splits = TraitorsEnvironment.list_splits()
        assert splits == ["train", "test"]

    def test_list_tasks_train(self):
        from thetraitors import TraitorsEnvironment
        tasks = TraitorsEnvironment.list_tasks("train")
        assert len(tasks) == 100

    def test_list_tasks_test(self):
        from thetraitors import TraitorsEnvironment
        tasks = TraitorsEnvironment.list_tasks("test")
        assert len(tasks) == 100

    def test_task_structure(self):
        from thetraitors import TraitorsEnvironment
        tasks = TraitorsEnvironment.list_tasks("train")
        for t in tasks:
            assert "id" in t
            assert "seed" in t
            assert "agent_role" in t
            assert "agent_player_index" in t
            assert t["agent_role"] in ("faithful", "traitor")
            assert 0 <= t["agent_player_index"] < 22

    def test_task_role_distribution(self):
        from thetraitors import TraitorsEnvironment
        tasks = TraitorsEnvironment.list_tasks("train")
        faithful_count = sum(1 for t in tasks if t["agent_role"] == "faithful")
        traitor_count = sum(1 for t in tasks if t["agent_role"] == "traitor")
        assert faithful_count == 50
        assert traitor_count == 50

    def test_no_seed_overlap(self):
        from thetraitors import TraitorsEnvironment
        train_seeds = {t["seed"] for t in TraitorsEnvironment.list_tasks("train")}
        test_seeds = {t["seed"] for t in TraitorsEnvironment.list_tasks("test")}
        assert len(train_seeds & test_seeds) == 0


# =============================================================================
# Run with: pytest golden_tests.py -v -k "engine" (for engine tests only)
# Run with: pytest golden_tests.py -v -k "NPC" (for NPC tests only)
# Run with: pytest golden_tests.py -v -k "Integration" (for integration tests)
# Run with: pytest golden_tests.py -v -k "Smoke" (for smoke tests)
# Run with: pytest golden_tests.py -v (for all tests)
# =============================================================================

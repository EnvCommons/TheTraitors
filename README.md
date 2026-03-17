# The Traitors

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/RJT1990/TheTraitors)

## Description

The Traitors is a social deduction environment based on the UK TV show "The Traitors." The agent plays as one of 22 contestants — either a Faithful or a Traitor — in a multi-round game of deception, discussion, and elimination. All 21 other players are NPCs driven by gpt-5-mini with persistent personalities, memories, and suspicion tracking.

Each round follows the show's format: a night phase (Traitors secretly murder a Faithful), breakfast (reveal), a team mission, a roundtable discussion, and a banishment vote. The game tests social reasoning, strategic voting, deception (as Traitor), and deduction (as Faithful).

## Capabilities

- Multi-agent social deduction with 22 players
- Strategic discussion and accusation at the roundtable
- Banishment voting with vote-pattern analysis
- Murder target selection and traitor coordination (Traitor role)
- Endgame decision-making under uncertainty
- Persistent NPC memory with suspicion and trust tracking

## Compute Requirements

No sandbox required. The environment runs entirely in-process. NPC decisions use gpt-5-mini API calls.

## Tasks

There are 100 training tasks and 100 test tasks.

- **Train split** (seeds 0-99): 50 tasks as Faithful, 50 as Traitor
- **Test split** (seeds 10000-10099): 50 tasks as Faithful, 50 as Traitor

Each task initialises a fresh 22-player game with deterministic player names, personalities, and role assignments based on the seed. Games last up to 15 rounds, typically ending in 8-12 rounds as players are eliminated.

## Reward Structure

This is a dense, verifiable reward environment. Rewards are computed from game state with no LLM graders.

| Component | Value | Condition |
|-----------|-------|-----------|
| Survival bonus | +0.02/round | Awarded each round the agent survives (capped at +0.2) |
| Correct traitor vote | +0.1/vote | Agent voted for a player who was banished and revealed as Traitor |
| Wrong faithful vote | -0.03/vote | Agent voted for a player who was banished and revealed as Faithful |
| Win condition | +0.5 | Agent's side wins AND agent is still alive |
| Prize pot share | up to +0.2 | Proportional to accumulated prize pot vs maximum possible |

Total reward is clamped to [0, 1].

## Tools

| Tool | Phase | Description |
|------|-------|-------------|
| `speak(message)` | Discussion | Make a statement at the roundtable |
| `view_game_state()` | Any | View alive players, prize pot, elimination history, voting records |
| `cast_vote(player_name)` | Roundtable Vote | Vote to banish a player |
| `nominate_murder(player_name)` | Night (Traitor only) | Choose a Faithful to murder |
| `discuss_with_traitors(message)` | Night (Traitor only) | Private discussion in the Traitor conclave |
| `vote_endgame(choice)` | Endgame Vote | Vote "end_game" or "banish_again" |

## Time Horizon

Games last 8-15 rounds. Each round involves multiple tool calls (discussion exchanges + vote). A typical game involves 30-60 agent tool calls.

## Environment Difficulty

The environment requires strategic social reasoning against 21 LLM-driven NPCs. Faithful agents must identify Traitors from discussion patterns, voting behaviour, and elimination history. Traitor agents must blend in, deflect suspicion, and coordinate murders without being caught.

## Other Environment Requirements

This environment requires an OpenAI API key passed via the `openai_api_key` secret for NPC decision-making (uses gpt-5-mini).

## Safety

Agents interact with a simulated social deduction game. The environment does not present direct safety risks. Agents make strategic decisions about accusation, voting, and deception within a well-defined game framework. The deception required of Traitor agents is bounded by the game context and does not transfer to real-world manipulation scenarios.

## Citations

```bibtex
@dataset{GRTheTraitors,
  author    = {General Reasoning Inc. Team},
  title     = {The Traitors},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/EnvCommons/thetraitors}
}
```

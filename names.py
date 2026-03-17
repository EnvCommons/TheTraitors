"""
Deterministic name and personality generation for The Traitors environment.
Uses seeded RNG to produce consistent player setups across runs.
"""

import random

# 35 British names (more than 22 needed to allow variety across seeds)
NAME_POOL = [
    "Alistair", "Beatrice", "Callum", "Diana", "Edmund",
    "Fiona", "Graham", "Helena", "Ian", "Judith",
    "Kenneth", "Lydia", "Malcolm", "Natasha", "Oliver",
    "Penelope", "Rupert", "Sophie", "Thomas", "Victoria",
    "William", "Yasmin", "Zachary", "Abigail", "Benjamin",
    "Charlotte", "Douglas", "Eleanor", "Freddie", "Georgia",
    "Hugo", "Isla", "James", "Katherine", "Liam",
]

# 25 personality templates (more than 22 needed)
PERSONALITY_TEMPLATES = [
    "a retired schoolteacher who is observant and methodical, always looking for logic in behaviour",
    "a young entrepreneur who is confident and occasionally confrontational, never afraid to speak up",
    "a nurse who is empathetic and reads people well, but can sometimes be too trusting",
    "a former police detective who is analytical and naturally suspicious of everyone",
    "a pub landlord who is loud, charismatic, and loves being the centre of attention",
    "a librarian who is quiet and watchful, only speaking up when something important needs saying",
    "a personal trainer who is direct and honest, sometimes blunt to a fault",
    "an actor who is dramatic and emotional, prone to grand gestures and passionate speeches",
    "a financial analyst who is calculating and strategic, always thinking several moves ahead",
    "a charity worker who believes deeply in fairness and gets genuinely upset by dishonesty",
    "a farmer who is plain-spoken, trusts their gut instinct, and doesn't suffer fools gladly",
    "a university lecturer who over-analyses everything and enjoys long debates about evidence",
    "a hairdresser who is chatty, sociable, and picks up on gossip and social dynamics quickly",
    "a software engineer who is logical to a fault and demands evidence for every single claim",
    "a baker who is warm and community-minded but fiercely protective of those they trust",
    "a retired army officer who is disciplined, strategic, and expects people to be straightforward",
    "a florist who is gentle and kind but surprisingly perceptive about people's true intentions",
    "a journalist who asks probing questions and is skilled at getting people to reveal things",
    "a bartender who has seen it all and can read a room better than anyone",
    "a primary school teacher who is patient and nurturing but has a strong sense of right and wrong",
    "a stand-up comedian who uses humour to deflect tension and often says what others are thinking",
    "a social worker who is deeply empathetic and always tries to see the best in everyone",
    "a property developer who is shrewd, business-minded, and always watching the bottom line",
    "a yoga instructor who is calm and centred, observing group dynamics with detached clarity",
    "a football coach who is competitive, strategic, and excellent at reading team dynamics",
]


def generate_player_setup(
    seed: int, agent_player_index: int, agent_role: str
) -> list[tuple[str, str, str]]:
    """
    Deterministically generate names, personalities, and roles for 22 players.

    Args:
        seed: Random seed for deterministic generation
        agent_player_index: Which player slot (0-21) the agent occupies
        agent_role: "faithful" or "traitor"

    Returns:
        List of 22 tuples: (name, personality, role)
        where role is "faithful" or "traitor"
    """
    rng = random.Random(seed)

    num_players = 22
    num_traitors = 3

    # Select 22 names from pool
    names = rng.sample(NAME_POOL, num_players)

    # Select 22 personalities from pool
    personalities = rng.sample(PERSONALITY_TEMPLATES, num_players)

    # Assign roles: agent's role is fixed, fill remaining traitor slots
    traitor_indices = set()
    if agent_role == "traitor":
        traitor_indices.add(agent_player_index)

    # Fill remaining traitor slots from non-agent positions
    available = [i for i in range(num_players) if i not in traitor_indices]
    rng.shuffle(available)

    needed = num_traitors - len(traitor_indices)
    for i in range(needed):
        traitor_indices.add(available[i])

    result = []
    for i in range(num_players):
        role = "traitor" if i in traitor_indices else "faithful"
        result.append((names[i], personalities[i], role))

    return result

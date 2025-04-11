import numpy as np
import itertools
import pandas as pd
import pulp
from mechanisms import probabilistic_serial, random_priority, popular_assignment
from typing import Dict, List, Tuple
import scipy.optimize
import networkx as nx

# Mapping mechanism names to the corresponding functions
MECHANISMS = {
    "Probabilistic Serial": probabilistic_serial,
    "Random Priority": random_priority,
    "Popular": popular_assignment
}

def check_ex_post_efficiency(
    assignment: Dict[str, Dict[str, float]],
    preferences: Dict[str, List[str]],
    agents: List[str],
    objects: List[str]
) -> bool:
    """
    Checks whether a random assignment is ex post efficient.

    Ex post efficiency holds if the assignment can be expressed as a convex combination 
    of deterministic Pareto optimal assignments.

    Args:
        assignment: A nested dictionary where assignment[agent][object] = probability.
        preferences: A dictionary mapping each agent to their ordered list of preferred objects.
        agents: List of agent names.
        objects: List of object names.

    Returns:
        True if the assignment is ex post efficient, False otherwise.
    """
    n = len(agents)
    perms = list(itertools.permutations(objects))
    
    def is_pareto_optimal(perm: Tuple[str, ...]) -> bool:
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i == j:
                    continue
                obj_i, obj_j = perm[i], perm[j]
                if preferences[agent_i].index(obj_j) < preferences[agent_i].index(obj_i) and \
                   preferences[agent_j].index(obj_i) < preferences[agent_j].index(obj_j):
                    return False
        return True

    pareto_perms = [perm for perm in perms if is_pareto_optimal(perm)]

    A_eq, b_eq = [], []
    for i, agent in enumerate(agents):
        for j, obj in enumerate(objects):
            row = [1.0 if perm[i] == obj else 0.0 for perm in pareto_perms]
            A_eq.append(row)
            b_eq.append(assignment[agent][obj])

    result = scipy.optimize.linprog(
        c=np.zeros(len(pareto_perms)),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=[(0, 1)] * len(pareto_perms),
        method="highs"
    )
    
    return result.success

def check_ordinal_efficiency(
    assignment: Dict[str, Dict[str, float]],
    preferences: Dict[str, List[str]],
    agents: List[str],
    objects: List[str]
) -> bool:
    """
    Checks whether a random assignment is ordinally efficient.

    Ordinal efficiency holds if the assignment is not stochastically dominated by 
    another assignment under agents' ordinal preferences. This is checked via the 
    absence of cycles in a preference-induced dominance graph.

    Args:
        assignment: A nested dictionary where assignment[agent][object] = probability.
        preferences: A dictionary mapping each agent to their ordered list of preferred objects.
        agents: List of agent names.
        objects: List of object names.

    Returns:
        True if the assignment is ordinally efficient, False if a dominance cycle is found.
    """
    G = nx.DiGraph()
    G.add_nodes_from(objects)

    for agent in agents:
        pref = preferences[agent]
        for i, better in enumerate(pref):
            for worse in pref[i + 1:]:
                if assignment[agent][worse] > 1e-6:
                    G.add_edge(better, worse)

    try:
        nx.find_cycle(G, orientation='original')
        return False
    except nx.exception.NetworkXNoCycle:
        return True



def check_no_envy(
    assignment: Dict[str, Dict[str, float]],
    preferences: Dict[str, List[str]],
    agents: List[str],
    objects: List[str]
) -> bool:
    """
    Verify if the assignment satisfies no envy.
    No envy means no agent prefers the assignment of another agent strictly.
    
    Args:
        assignment: A dictionary where each agent (str) maps to another dictionary mapping each object (str) to a fraction (float).
        preferences: A dictionary where each agent (str) maps to a list of objects (str) ranked by preference, in order.
        agents: A list of agent names (strings).
        objects: A list of object names (strings).

    Returns:
        A boolean indicating whether the assignment satisfies the no-envy property.
    """
    for agent1 in agents:
        for agent2 in agents:
            if agent1 != agent2:
                agent1_utility = sum(
                    (len(objects) - preferences[agent1].index(obj)) * assignment[agent1][obj]
                    for obj in assignment[agent1]
                )
                agent2_utility = sum(
                    (len(objects) - preferences[agent1].index(obj)) * assignment[agent2][obj]
                    for obj in assignment[agent2]
                )
                if agent2_utility > agent1_utility:
                    return False  # Found envy
    return True


def check_strategy_proofness(
    assignment: Dict[str, Dict[str, float]],
    preferences: Dict[str, List[str]],
    agents: List[str],
    objects: List[str],
    mechanism_name: str
) -> bool:
    """
    Verify if the assignment is strategy-proof.
    Strategy-proof means that agents cannot benefit from misreporting their preferences.
    
    Args:
        assignment: A dictionary where each agent (str) maps to another dictionary mapping each object (str) to a fraction (float).
        preferences: A dictionary where each agent (str) maps to a list of objects (str) ranked by preference, in order.
        agents: A list of agent names (strings).
        objects: A list of object names (strings).
        mechanism_name: A string indicating the name of the mechanism being tested.

    Returns:
        A boolean indicating whether the assignment is strategy-proof.
    """
    # Get the mechanism function based on the name
    mechanism = MECHANISMS.get(mechanism_name)
    if mechanism is None:
        raise ValueError(f"Unknown mechanism: {mechanism_name}")

    for agent in agents:
        original_preferences = preferences[agent]
        for i in range(len(objects) - 1):
            # Simulate misreporting by swapping preferences
            misreported_preferences = original_preferences[:]
            misreported_preferences[i], misreported_preferences[i + 1] = (
                misreported_preferences[i + 1], misreported_preferences[i]
            )
            preferences[agent] = misreported_preferences

            # Recalculate assignment with misreported preferences
            new_assignment = mechanism(agents, objects, preferences)

            # Compare utility
            original_utility = sum(
                (len(objects) - original_preferences.index(obj)) * assignment[agent][obj]
                for obj in assignment[agent]
            )
            new_utility = sum(
                (len(objects) - original_preferences.index(obj)) * new_assignment[agent][obj]
                for obj in new_assignment[agent]
            )
            if new_utility > original_utility:
                return False  # Found an improvement with misreporting, not strategy-proof

        # Restore original preferences
        preferences[agent] = original_preferences

    return True


def evaluate_mechanism_properties(
    agents: List[str],
    objects: List[str],
    preferences: Dict[str, List[str]],
    mechanism: str
) -> Dict[str, bool]:
    """
    Evaluate all properties (Ex Post Efficiency, Ordinal Efficiency, No Envy, Strategy-Proofness)
    for a given mechanism.

    Args:
        agents: A list of agent names (strings).
        objects: A list of object names (strings).
        preferences: A dictionary where each agent (str) maps to a list of objects (str) ranked by preference, in order.
        mechanism: A string specifying the mechanism to evaluate.

    Returns:
        A dictionary where the keys are the names of the properties (str) and the values are booleans indicating if the property holds.
    """
    # Generate the assignment using the given mechanism
    if mechanism == "Probabilistic Serial":
        assignment = probabilistic_serial(agents, objects, preferences)
    elif mechanism == "Random Priority":
        assignment = random_priority(agents, objects, preferences)
    elif mechanism == "Popular":
        assignment = popular_assignment(agents, objects, preferences)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    # Evaluate properties
    ex_post_eff = check_ex_post_efficiency(assignment, preferences, agents, objects)
    ordinal_eff = check_ordinal_efficiency(assignment, preferences, agents, objects)
    no_envy = check_no_envy(assignment, preferences, agents, objects)
    strategy_proof = check_strategy_proofness(assignment, preferences, agents, objects, mechanism)

    # Return the results as a dictionary
    return {
        "Ex Post Efficiency": ex_post_eff,
        "Ordinal Efficiency": ordinal_eff,
        "No Envy": no_envy,
        "Strategy-Proofness": strategy_proof
    }

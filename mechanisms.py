import numpy as np
import itertools
import pulp
from typing import Dict, List

def probabilistic_serial(
    agents: List[str],
    objects: List[str],
    preferences: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Implement the Probabilistic Serial mechanism for dessert allocation.
    
    Args:
        agents: A list of agent names (strings).
        objects: A list of object names (strings).
        preferences: A dictionary where each agent (str) maps to a list of objects (str) ranked by preference, in order.

    Returns:
        A dictionary where each agent maps to a dictionary of object allocations (fractions).
    """
    assignment = {a: {o: 0.0 for o in objects} for a in agents}
    remaining = {o: 1.0 for o in objects}
    time = 0.0

    while time < 1.0 and remaining:
        eat_rates = {}
        for agent in agents:
            for obj in preferences[agent]:
                if obj in remaining and remaining[obj] > 1e-6:
                    eat_rates[agent] = obj
                    break

        if not eat_rates:
            break

        delta = 1.0 - time
        consumed_objects = set(eat_rates.values())
        for obj in consumed_objects:
            agents_eating = [a for a, o in eat_rates.items() if o == obj]
            rate = len(agents_eating)
            obj_delta = remaining[obj] / rate
            delta = min(delta, obj_delta)

        for agent, obj in eat_rates.items():
            if obj in remaining:
                assignment[agent][obj] += delta
                remaining[obj] -= delta
                if remaining[obj] < 1e-6:
                    del remaining[obj]

        time += delta

    return assignment


def random_priority(
    agents: List[str],
    objects: List[str],
    preferences: Dict[str, List[str]],
    iterations: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Implement the Random Priority mechanism for dessert allocation.
    
    Args:
        agents: A list of agent names (strings).
        objects: A list of object names (strings).
        preferences: A dictionary where each agent (str) maps to a list of objects (str) ranked by preference.
        iterations: The number of random simulations to run (default 1000).

    Returns:
        A dictionary where each agent maps to a dictionary of object allocations (fractions).
    """
    assignment = {a: {o: 0.0 for o in objects} for a in agents}

    for _ in range(iterations):
        order = np.random.permutation(agents)
        remaining = objects.copy()
        temp_assign = {}

        for agent in order:
            for obj in preferences[agent]:
                if obj in remaining:
                    temp_assign[agent] = obj
                    remaining.remove(obj)
                    break

        for a, o in temp_assign.items():
            assignment[a][o] += 1 / iterations

    return assignment


def popular_assignment(
    agents: List[str],
    objects: List[str],
    preferences: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Implement the Popular mechanism for dessert allocation using linear programming.
    
    Args:
        agents: A list of agent names (strings).
        objects: A list of object names (strings).
        preferences: A dictionary where each agent (str) maps to a list of objects (str) ranked by preference.

    Returns:
        A dictionary where each agent maps to a dictionary of object allocations (fractions).
    """
    n = len(agents)
    prob = pulp.LpProblem("PopularAssignment", pulp.LpMaximize)

    p = pulp.LpVariable.dicts("p", (agents, objects), lowBound=0, upBound=1)
    z = pulp.LpVariable("z", lowBound=-n, upBound=n)

    prob += z

    for agent in agents:
        prob += pulp.lpSum([p[agent][o] for o in objects]) == 1
    for obj in objects:
        prob += pulp.lpSum([p[agent][obj] for agent in agents]) == 1

    all_perms = list(itertools.permutations(objects))
    for M in all_perms:
        margin = 0
        for i, agent in enumerate(agents):
            m_obj = M[i]
            for o in objects:
                if preferences[agent].index(o) < preferences[agent].index(m_obj):
                    margin += p[agent][o]
                elif preferences[agent].index(o) > preferences[agent].index(m_obj):
                    margin -= p[agent][o]
        prob += margin >= z

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {a: {o: p[a][o].varValue for o in objects} for a in agents}

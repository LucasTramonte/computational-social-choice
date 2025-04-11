import streamlit as st
import pandas as pd
from mechanisms import probabilistic_serial, random_priority, popular_assignment
from visualization import plot_heatmap
from verify_properties import (
    check_ex_post_efficiency,
    check_ordinal_efficiency,
    check_no_envy,
    check_strategy_proofness,
    evaluate_mechanism_properties
)
from typing import Dict, List

def dynamic_paragraph_interpretation(
    assignment: Dict[str, Dict[str, float]],
    agents: List[str],
    objects: List[str],
    mechanism: str
) -> None:
    """
    Generates and displays a dynamic explanation of the dessert assignment for a given mechanism.
    
    Args:
        assignment: A dictionary where each agent maps to another dictionary that maps objects to the assigned fractions (float).
        agents: A list of agent names (strings).
        objects: A list of object names (strings).
        mechanism: The mechanism used for the assignment (string), one of 'Probabilistic Serial', 'Random Priority', or 'Popular'.
    
    Returns:
        None
    """
    paragraph = ""

    if mechanism == 'Probabilistic Serial':
        paragraph += "The Probabilistic Serial mechanism involves agents starting with their most preferred dessert and gradually moving on to others. The portions they get depend on how much time is allocated to each dessert. Each agent 'eats' part of the dessert in order of preference, with some getting more or less depending on the progress of time.\n\n"
    elif mechanism == 'Random Priority':
        paragraph += "The Random Priority mechanism means the order in which agents select their desserts is random. Each agent gets a portion of their preferred dessert depending on the random selection order across multiple simulations. The percentages reflect the frequency with which each dessert was allocated to an agent over several trials.\n\n"
    elif mechanism == 'Popular':
        paragraph += "The Popular mechanism distributes desserts based on collective preferences, ensuring the most popular desserts are allocated according to the majority's wishes. This results in allocations that reflect what is most favored by the group as a whole, balancing fairness and collective preference.\n\n"

    # Per-agent explanation
    for agent in agents:
        portions = {obj: assignment[agent].get(obj, 0.0) * 100 for obj in objects}
        
        paragraph += f"For {agent}, the results show that they received "
        portion_strings = [f"{portions[obj]:.2f}% of {obj}" for obj in objects]
        paragraph += ", ".join(portion_strings) + ".\n\n"

    # Common explanation after the agent-specific details
    if mechanism == 'Probabilistic Serial':
        paragraph += "This means that the agent's allocation is based on their preferences, with a gradual 'eating' process that allows for more or less of each dessert depending on timing and other agents' choices.\n\n"
    elif mechanism == 'Random Priority':
        paragraph += "This means that the allocation depends on the random order in which agents were selected, with the percentages showing how often the agent received each dessert across multiple random simulations.\n\n"
    elif mechanism == 'Popular':
        paragraph += "This indicates that the allocation was determined by the collective preferences, and the percentages represent the proportion of each dessert the agent received based on majority preferences.\n\n"

    # Displaying the dynamic interpretation in Streamlit
    st.markdown(paragraph)

# Streamlit UI
st.title("Dessert Assignment Mechanism Comparison")

st.sidebar.header("Customize Your Preferences")
n = st.sidebar.slider("Number of agents / objects", min_value=2, max_value=6, value=3)
agents = [st.sidebar.text_input(f"Agent {i+1} name", value=f"Agent {i+1}") for i in range(n)]
objects = [st.sidebar.text_input(f"Object {i+1} name", value=f"Éclair au chocolat" if i == 0 else ("Tarte aux pommes" if i == 1 else "Mousse au citron")) for i in range(n)]

st.markdown("### Agents' Preferences (Ranked List of Desserts)")
preferences: Dict[str, List[str]] = {}
for agent in agents:
    pref = st.multiselect(f"Preferences of {agent}", objects, default=objects, key=agent)
    if len(pref) == n:
        preferences[agent] = pref
    else:
        st.warning(f"{agent} must rank all objects exactly once.")

if len(preferences) == n:
    mechanism = st.selectbox("Choose a mechanism", ['Random Priority', 'Probabilistic Serial', 'Popular'])

    with st.spinner('Calculating...'):
        if mechanism == 'Random Priority':
            assignment = random_priority(agents, objects, preferences)
        elif mechanism == 'Probabilistic Serial':
            assignment = probabilistic_serial(agents, objects, preferences)
        else:
            assignment = popular_assignment(agents, objects, preferences)

    st.subheader(f"Assignment Result - {mechanism}")
    df = pd.DataFrame(assignment).T
    plot_heatmap(df, mechanism)
    dynamic_paragraph_interpretation(assignment, agents, objects, mechanism)
    
    # Verify properties using the new function
    st.subheader("Property Verification")
    properties = evaluate_mechanism_properties(agents, objects, preferences, mechanism)

    st.markdown(f"**Ex Post Efficiency:** {'✅ Yes' if properties['Ex Post Efficiency'] else '❌ No'}")
    st.markdown(f"**Ordinal Efficiency:** {'✅ Yes' if properties['Ordinal Efficiency'] else '❌ No'}")
    st.markdown(f"**No Envy:** {'✅ Yes' if properties['No Envy'] else '❌ No'}")
    st.markdown(f"**Strategy-Proofness:** {'✅ Yes' if properties['Strategy-Proofness'] else '❌ No'}")
    

    st.download_button("Download Results (CSV)", df.to_csv(index=False), file_name="assignment_results.csv")

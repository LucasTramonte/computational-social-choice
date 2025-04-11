# computational-social-choice
 Implements and compares various computational social choice mechanisms for fair allocation of resources

 [[User-friendly app]](https://lucastramonte-computational-social-choice-app-6tid8e.streamlit.app/)

## Overview

This project implements and analyzes several random assignment mechanisms for indivisible goods (e.g., desserts), based on agents' ordinal preferences. It includes:

- Algorithmic implementations of standard assignment rules
- Formal verification of economic properties (efficiency, fairness, strategy-proofness)
- A user-friendly interface built with **Streamlit** for interactive exploration

## Implemented Mechanisms

1. **Random Priority (RP)** – also known as Random Serial Dictatorship
2. **Probabilistic Serial (PS)** – based on the continuous eating algorithm
3. **Popular Assignment (PA)** – based on majority preference and stochastic dominance

## Project Structure

- `app.py` – main Streamlit app
- `mechanisms.py` – core implementations of the assignment algorithms
- `verify_properties.py` – functions to check efficiency, fairness, and strategy-proofness
- `visualization.py` – utility for displaying assignment heatmaps
- `requirements.txt` – list of required Python packages

## Theoretical Properties Verified

Each mechanism is evaluated for the following properties:

- **Ex Post Efficiency** – can the outcome be expressed as a lottery over Pareto-optimal deterministic assignments?
- **Ordinal Efficiency** – is there no other assignment that all agents weakly prefer and at least one strictly prefers?
- **No Envy** – does any agent prefer another agent's allocation to their own?
- **Strategy-Proofness** – can an agent gain from misreporting their preferences?

When a property fails, the app provides a concrete counterexample using the actual assignment values.

## How to Run

1. Clone the repository
2. Run `pip install -r requirements.txt`
3. Launch the app with `streamlit run app.py`
4. Open your browser to `http://localhost:8501`

## References

- A. Bogomolnaia and H. Moulin. "A New Solution to the Random Assignment Problem." *Journal of Economic Theory*, 2001.
- H. Aziz, F. Brandt, and P. Stursberg. "On Popular Random Assignments." *COMSOC*, 2013.

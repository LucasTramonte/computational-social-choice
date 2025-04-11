import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    cmap: str = 'viridis'
) -> None:
    """
    Plots a heatmap for a given DataFrame and displays it using Streamlit.

    Args:
        df: A pandas DataFrame containing the values to be visualized in the heatmap.
        title: A string representing the title of the heatmap.
        cmap: A string specifying the colormap to use for the heatmap. Default is 'viridis'.
    
    Returns:
        None: The function uses Streamlit to display the plot.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(df.values, cmap=cmap, vmin=0, vmax=1)

    # Set the ticks for x and y axes
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    # Rotate the x-axis labels for readability
    plt.xticks(rotation=45, ha='right')

    # Add text annotations inside the heatmap cells
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            text = ax.text(
                j, i, f"{df.iloc[i, j]:.2f}", ha='center', va='center',
                color='white' if df.iloc[i, j] < 0.5 else 'black'
            )

    # Display the plot in Streamlit
    st.pyplot(fig)

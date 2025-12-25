import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Define the input folder and file
input_folder = "data"
input_file = "death_pool_stats.csv"
input_path = os.path.join(input_folder, input_file)
normalized_input_path = os.path.normpath(input_path)

# Define the output folder and file
output_folder = "images"
output_file = "death_pool_standings.png"
os.makedirs(output_folder, exist_ok=True)
output_image = os.path.join(output_folder, output_file)
normalized_output_image = os.path.normpath(output_image)

# Define number of columns for subplots
cols = 4


def plot_cumulative_years(ax, year_range, df):
    """
    Generate a scatter plot for a cumulative year range.
    - Total Deaths (x-axis)
    - Total Points (y-axis)
    - Total Wins (size of point)
    - Average Rank (color of point: blue = best, red = worst)
    """
    # Filter the DataFrame for the specified year range
    df_filtered = df[df["Year"].isin(year_range)]

    # Aggregate data by player
    df_player = df_filtered.groupby("Player").agg(
        Total_Deaths=("Deaths", "sum"),
        Total_Points=("Points", "sum"),
        Total_Wins=("Wins", "sum"),
        Average_Rank=("Rank", "mean"),
        Count_Years=("Year", "nunique")
    ).reset_index()

    # Create scatter plot
    scatter = ax.scatter(
        df_player["Total_Deaths"],
        df_player["Total_Points"],
        c=df_player["Average_Rank"],  # Normalized rank for color mapping
        s=(np.clip(2 ** df_player["Total_Wins"], None, 500)) * 100,  # Exponential size scaling with cap at 500
        cmap="coolwarm",  # Red to blue color map
        alpha=0.7  # Transparency for better visibility
    )

    # Set titles and labels using the last year in the range
    ax.set_xlabel("Deaths")
    ax.set_ylabel("Points")
    ax.set_title(f"Death Pool Standings ({year_range[-1]})")

    # Add color bar for average rank
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Avg. Rank")

    # Annotate each point with the player's name
    for _, row in df_player.iterrows():
        ax.text(
            row["Total_Deaths"], row["Total_Points"], row["Player"],
            fontsize=8, ha="right", va="bottom", alpha=0.7
        )

def main():
    # Read the CSV data
    df = pd.read_csv(input_path)

    # Validate required columns
    required_columns = ["Deaths", "Points", "Wins", "Rank", "Player", "Year"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Normalize Rank for color mapping
    df["Rank_Color"] = df["Rank"].apply(
        lambda x: (x - df["Rank"].min()) / (df["Rank"].max() - df["Rank"].min())
    )

    # Generate year ranges for subplots
    valid_years = sorted(df["Year"].unique())
    year_ranges = [list(range(min(valid_years), y + 1)) for y in valid_years]

    # Calculate number of rows needed for subplots
    rows = (len(year_ranges) + cols - 1) // cols

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Generate plots for each year range
    for i, year_range in enumerate(year_ranges):
        plot_cumulative_years(axes[i], year_range, df)

        # Customize grid and ticks
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    # Turn off unused subplots
    for ax in axes[len(year_ranges):]:
        ax.axis("off")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save and show the final figure
    plt.savefig(normalized_output_image, dpi=300)
    plt.show()
    print(f"Visualization saved as {normalized_output_image}")


if __name__ == "__main__":
    main()

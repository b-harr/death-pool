import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator

# Define the path to the data folder and the file
data_folder = "data"
data_file = "death_pool_stats.csv"

# Construct the full file path
data_path = os.path.join(data_folder, data_file)

# Load the dataset
df = pd.read_csv(data_path)

# Data validation: Check for missing columns
required_columns = ["Deaths", "Points", "Wins", "Rank", "Player", "Year"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Normalize 'Average Rank' for coloring once, outside the loop
df["Rank_Color"] = df["Rank"].apply(
    lambda x: (x - df["Rank"].min()) / (df["Rank"].max() - df["Rank"].min())
)

# Function to generate the plot for each cumulative year range
def plot_cumulative_years(ax, year_range, df):
    """
    Generate a scatter plot for a specific year range showing player stats:
    - Deaths
    - Points
    - Wins
    - Average Rank
    
    The plot uses a color map to represent average rank and adjusts point sizes based on wins.
    """
    # Filter data for the given year range
    df_filtered = df[df["Year"].isin(year_range)]

    # Aggregate the data for each player
    df_player = df_filtered.groupby("Player").agg(
        Total_Deaths=("Deaths", "sum"),
        Total_Points=("Points", "sum"),
        Total_Wins=("Wins", "sum"),
        Average_Rank=("Rank", "mean"),
        Count_Years=("Year", "nunique")
    ).reset_index()

    # Plot the data in the provided axis (ax)
    scatter = ax.scatter(
        df_player["Total_Deaths"],
        df_player["Total_Points"],
        c=df_player["Average_Rank"],  # Use normalized rank for color
        s=(np.clip(2 ** df_player["Total_Wins"], None, 500)) * 100,  # Exponential scaling for size, capped at 500
        cmap="coolwarm",  # Use 'coolwarm' colormap for coloring based on rank
        alpha=0.7  # Make the dots semi-transparent
    )

    # Set labels and title (use the last year in the range for the title)
    ax.set_title(f"Death Pool Standings ({year_range[-1]})")
    ax.set_xlabel("Deaths")
    ax.set_ylabel("Points")

    # Add a colorbar to the plot for rank indication
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Avg. Rank")

    # Label each point with the player name, without background shading
    for _, row in df_player.iterrows():
        ax.text(
            row["Total_Deaths"], row["Total_Points"], row["Player"],
            fontsize=8, ha="right", va="bottom", alpha=0.7
        )

# Define the year ranges (cumulative, from 2017 to 2024)
year_ranges = [list(range(2017, year + 1)) for year in range(2017, 2026)]

# Number of subplots needed
num_plots = len(year_ranges)
cols = 4  # Number of columns of subplots
rows = (num_plots + cols - 1) // cols  # Calculate the number of rows based on the number of year ranges

# Create a figure with subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 3 * rows))

# Flatten axes array for easier iteration
axes = axes.flatten()

# Label each subplot with the corresponding year range
for i, year_range in enumerate(year_ranges):
    # Plot for the current year range
    plot_cumulative_years(axes[i], year_range, df)
        
    # Add a grid for better visualization
    axes[i].grid(True, linestyle="--", alpha=0.5)

    # Force integer ticks for both x and y axes
    axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

# Hide unused axes if there are any extra subplots
for ax in axes[len(year_ranges):]:
    ax.axis("off")

# Adjust layout to avoid overlapping subplots
plt.tight_layout()

# Ensure the 'images' folder exists for saving the plot
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# Save the plot as a high-resolution image in the 'images' folder
output_image = os.path.join(output_folder, "death_pool_standings.png")
plt.savefig(output_image, dpi=300)

# Display the plot
plt.show()

# Print a completion message with the output path
print(f"Visualization saved as {output_image}")

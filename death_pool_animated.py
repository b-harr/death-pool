import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
import os

# Function to safely load the dataset and handle errors if file is missing or invalid
def load_data(file_path):
    try:
        # Try to load the dataset from the specified CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns are present in the dataset
        if "Year" not in df.columns or "Player" not in df.columns:
            raise ValueError("Missing required columns in the dataset.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit()  # Exit the script if file is not found
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the CSV file.")
        exit()  # Exit if there is a parsing error
    except ValueError as e:
        print(e)  # Print the error message if required columns are missing
        exit()  # Exit if required columns are missing

# Load the dataset with error handling
df = load_data("data/death_pool_stats.csv")

# Dynamically determine the valid year range from the dataset
valid_years = df["Year"].unique()  # Get the unique years in the dataset
# Create a list of year ranges up to each valid year, ensuring no years are skipped
year_ranges = [list(range(min(valid_years), year + 1)) for year in valid_years if year >= min(valid_years)]

# Function to generate a plot for a given year range and save it as an image
def plot_for_year(year_range, df, output_path):
    try:
        # Filter data for the given year range
        df_filtered = df[df["Year"].isin(year_range)]
        
        # Aggregate data for each player (sum of deaths, points, wins, and average rank)
        df_player = df_filtered.groupby("Player").agg(
            Total_Deaths=("Deaths", "sum"),
            Total_Points=("Points", "sum"),
            Total_Wins=("Wins", "sum"),
            Average_Rank=("Rank", "mean")
        ).reset_index()  # Reset index to have a clean DataFrame
        
        # Normalize the 'Average Rank' for coloring the points on the plot
        df_player["Rank_Color"] = df_player["Average_Rank"].apply(
            lambda x: (x - df_player["Average_Rank"].min()) / (df_player["Average_Rank"].max() - df_player["Average_Rank"].min())
        )
        
        # Adjust the scaling of the dots based on the number of wins (size of each dot)
        sizing_factor = 100  # Base size for 0 wins
        max_size = 500  # Maximum size for players with 2 wins

        # Scale the size of the dots based on the number of wins
        df_player["Size"] = (df_player["Total_Wins"] / 2) * (max_size - sizing_factor) + sizing_factor

        # Create the plot with adjusted sizes and color coding
        fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size for the plot
        scatter = ax.scatter(
            df_player["Total_Deaths"],  # X-axis: Total deaths
            df_player["Total_Points"],  # Y-axis: Total points
            c=df_player["Rank_Color"],  # Color points based on average rank
            s=df_player["Size"],        # Size of points based on number of wins
            cmap="coolwarm",            # Use coolwarm colormap for ranking
            alpha=0.7                   # Set transparency for better visualization
        )
        
        # Set the title and labels for the plot
        ax.set_title(f"Death Pool Standings ({year_range[-1]})")  # Use the last year in the range for title
        ax.set_xlabel("Deaths")
        ax.set_ylabel("Points")
        
        # Force integer ticks on both axes for better readability
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add a grid for better visualization
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add a colorbar to show the average rank color scale
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Avg. Rank")
        
        # Label each point with the player's name for clarity
        for _, row in df_player.iterrows():
            ax.text(
                row["Total_Deaths"], row["Total_Points"], row["Player"],  # Position of the label
                fontsize=8, ha="right", va="bottom", alpha=0.7           # Label appearance
            )
        
        # Adjust layout to avoid clipping and save the plot as an image
        plt.tight_layout()  # Adjust layout for a better fit
        plt.savefig(output_path, format="png")  # Save the plot as a PNG image
        plt.close(fig)  # Close the figure to free up memory
    except Exception as e:
        print(f"Error generating plot for year range {year_range}: {e}")  # Handle any errors that occur during plot generation

# Check if the "images" directory exists, if not, create it
if not os.path.exists("images"):
    os.makedirs("images")

# List to store paths of each generated frame (image) for the GIF
frame_paths = []

# Generate plots for each year range and save them as individual images
for year_range in year_ranges:
    output_path = f"images/frame_{year_range[-1]}.png"  # Define the output path for the image
    plot_for_year(year_range, df, output_path)  # Generate the plot and save it as an image
    frame_paths.append(output_path)  # Add the path of the saved image to the list of frames

# Create an animated GIF from the generated plot images
try:
    # Open each frame image and create the animated GIF
    frames = [Image.open(frame) for frame in frame_paths]
    frames[0].save(
        "images/death_pool_standings.gif",  # Output path for the GIF
        save_all=True,                      # Save multiple frames to the GIF
        append_images=frames[1:],           # Append all frames except the first one
        duration=2500,                      # Duration of each frame in the GIF (2.5 seconds)
        loop=0                              # Loop indefinitely (0 means infinite loop)
    )
    print("Animated GIF saved as 'images/death_pool_standings.gif'.")
except Exception as e:
    print(f"Error creating animated GIF: {e}")  # Handle any errors during GIF creation

# Clean up temporary frame images after the GIF is created to save disk space
for frame in frame_paths:
    try:
        os.remove(frame)  # Remove each temporary frame image
    except Exception as e:
        print(f"Error removing frame {frame}: {e}")  # Handle any errors during frame removal

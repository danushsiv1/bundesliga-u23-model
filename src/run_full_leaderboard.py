from pathlib import Path
import pandas as pd
import sys

# Import the model and functions from your existing script
from projection_model import build_leaderboard

def main():
    # Prompt the user interactively if no command-line argument is provided
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = input("Enter the path to your CSV file [default: all_players_cleaned.csv]: ").strip()
        if not input_csv:
            input_csv = "all_players_cleaned.csv"
        
    print(f"\nRunning U23 Projection Model for all player pools using '{input_csv}'...")
    
    # Generate the leaderboard and the full dataset
    leaderboard, full_dataset = build_leaderboard(input_csv)
    
    # Ensure the output directory exists locally
    Path("output").mkdir(parents=True, exist_ok=True)
    
    # Save the complete leaderboard of all players to a CSV file
    output_path = "output/u23_projection_leaderboard.csv"
    leaderboard.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully ranked {len(leaderboard)} players!")
    print(f"Saved full leaderboard CSV to: {output_path}\n")

if __name__ == "__main__":
    main()
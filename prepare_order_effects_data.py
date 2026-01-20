"""
Prepare Order Effects Data for R Analysis

This script:
1. Reads the participant tracker CSV
2. Parses Game Order to determine order_position for Games A and B
3. Reads game JSON files to extract num_moves
4. Outputs a clean CSV for R analysis (two-way mixed ANOVA)

Output: order_effects_data.csv in Downloads folder
"""

import json
import pandas as pd
from pathlib import Path
from tkinter import Tk, filedialog

# Output directory
OUTPUT_DIR = "/Users/rachelpapirmeister/Downloads"


def select_file(title="Select a file", filetypes=None):
    """Open a file selection dialog"""
    if filetypes is None:
        filetypes = [("All files", "*.*")]
    Tk().withdraw()
    file = filedialog.askopenfilename(title=title, initialdir=OUTPUT_DIR, filetypes=filetypes)
    return file if file else None


def select_folder(title="Select a folder"):
    """Open a folder selection dialog"""
    Tk().withdraw()
    folder = filedialog.askdirectory(title=title, initialdir=OUTPUT_DIR)
    return folder if folder else None


def parse_game_order(order_string):
    """
    Parse game order string like "C, A, B" into position dict
    Returns: {'A': 2, 'B': 3, 'C': 1} (1-indexed positions)
    """
    if not order_string or pd.isna(order_string):
        return {}

    # Clean and split
    order_string = str(order_string).strip()
    games = [g.strip() for g in order_string.split(',')]

    # Create position dict (1-indexed)
    positions = {}
    for i, game in enumerate(games):
        positions[game] = i + 1

    return positions


def count_moves_in_json(json_path):
    """
    Read a game JSON file and count the number of movements
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Try different possible keys for movements
        if 'movements' in data:
            return len(data['movements'])
        elif 'moves' in data:
            return len(data['moves'])
        elif 'events' in data:
            # Count only movement events
            return len([e for e in data['events'] if e.get('type') == 'move'])
        else:
            print(f"  Warning: No movements found in {json_path}")
            return None
    except Exception as e:
        print(f"  Error reading {json_path}: {e}")
        return None


def find_game_file(filename, game_data_folder):
    """
    Find the actual game file in the folder
    Handles cases where filename might have extra info like "(mehdi)"
    """
    if not filename or filename == "--" or pd.isna(filename):
        return None

    # Clean the filename - remove parenthetical notes like "(mehdi)"
    clean_name = str(filename).strip()
    if '(' in clean_name:
        clean_name = clean_name.split('(')[0].strip()

    # Try exact match first
    full_path = Path(game_data_folder) / clean_name
    if full_path.exists():
        return full_path

    # Try finding by partial match (timestamp-based)
    if 'game-session' in clean_name or 'puzzle-game' in clean_name:
        # Extract the timestamp part
        for file in Path(game_data_folder).glob('*.json'):
            if clean_name in file.name or file.name in clean_name:
                return file

    # Try glob pattern
    pattern = clean_name.replace('.json', '*.json')
    matches = list(Path(game_data_folder).glob(pattern))
    if matches:
        return matches[0]

    return None


def main():
    print("=" * 60)
    print("PREPARE ORDER EFFECTS DATA")
    print("=" * 60)

    # Select participant tracker CSV
    print("\nPlease select your Participant Tracker CSV file...")
    tracker_file = select_file(
        "Select Participant Tracker CSV",
        [("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not tracker_file:
        print("No file selected. Exiting.")
        return
    print(f"Selected: {tracker_file}")

    # Select game data folder
    print("\nPlease select the folder containing game JSON files...")
    game_folder = select_folder("Select Game Data Folder")
    if not game_folder:
        print("No folder selected. Exiting.")
        return
    print(f"Selected: {game_folder}")

    # Read participant tracker
    print("\nReading participant tracker...")
    df = pd.read_csv(tracker_file)

    # Find the relevant columns (handle variations in column names)
    columns = df.columns.tolist()

    # Identify columns by position or name
    session_id_col = None
    game_order_col = None
    game_a_col = None
    game_b_col = None

    for i, col in enumerate(columns):
        col_lower = str(col).lower()
        if 'session' in col_lower and 'id' in col_lower:
            session_id_col = col
        elif 'game order' in col_lower or 'order' in col_lower:
            game_order_col = col
        elif 'game a' in col_lower and 'data' in col_lower:
            game_a_col = col
        elif 'game b' in col_lower and 'data' in col_lower:
            game_b_col = col

    # Fallback to positional if not found
    if not session_id_col:
        session_id_col = columns[1] if len(columns) > 1 else None
    if not game_order_col:
        game_order_col = columns[4] if len(columns) > 4 else None
    if not game_a_col:
        game_a_col = columns[8] if len(columns) > 8 else None
    if not game_b_col:
        game_b_col = columns[11] if len(columns) > 11 else None

    print(f"\nIdentified columns:")
    print(f"  Session ID: {session_id_col}")
    print(f"  Game Order: {game_order_col}")
    print(f"  Game A Data: {game_a_col}")
    print(f"  Game B Data: {game_b_col}")

    # Process each participant
    results = []

    print("\nProcessing participants...")
    for idx, row in df.iterrows():
        participant_id = str(row.get(session_id_col, '')).strip()

        # Skip empty rows
        if not participant_id or pd.isna(row.get(session_id_col)):
            continue

        # Skip excluded participants
        if '(exclude)' in participant_id.lower() or 'exclude' in participant_id.lower():
            print(f"  Skipping {participant_id} (excluded)")
            continue

        # Get game order
        game_order = row.get(game_order_col, '')
        positions = parse_game_order(game_order)

        if 'A' not in positions or 'B' not in positions:
            print(f"  Skipping {participant_id} - invalid game order: {game_order}")
            continue

        # Get file names
        game_a_file = row.get(game_a_col, '')
        game_b_file = row.get(game_b_col, '')

        # Process Game A
        if game_a_file and game_a_file != "--" and not pd.isna(game_a_file):
            game_a_path = find_game_file(game_a_file, game_folder)
            if game_a_path:
                num_moves_a = count_moves_in_json(game_a_path)
                if num_moves_a is not None:
                    results.append({
                        'participant_id': participant_id,
                        'game_type': 'Game A',
                        'order_position': positions['A'],
                        'num_moves': num_moves_a,
                        'game_file': str(game_a_path.name)
                    })
                    print(f"  {participant_id} Game A: position={positions['A']}, moves={num_moves_a}")
            else:
                print(f"  {participant_id} Game A: file not found - {game_a_file}")
        else:
            print(f"  {participant_id} Game A: no data file")

        # Process Game B
        if game_b_file and game_b_file != "--" and not pd.isna(game_b_file):
            game_b_path = find_game_file(game_b_file, game_folder)
            if game_b_path:
                num_moves_b = count_moves_in_json(game_b_path)
                if num_moves_b is not None:
                    results.append({
                        'participant_id': participant_id,
                        'game_type': 'Game B',
                        'order_position': positions['B'],
                        'num_moves': num_moves_b,
                        'game_file': str(game_b_path.name)
                    })
                    print(f"  {participant_id} Game B: position={positions['B']}, moves={num_moves_b}")
            else:
                print(f"  {participant_id} Game B: file not found - {game_b_file}")
        else:
            print(f"  {participant_id} Game B: no data file")

    # Create output dataframe
    if not results:
        print("\nNo data was processed. Check your files and try again.")
        return

    output_df = pd.DataFrame(results)

    # Save to CSV
    output_file = f"{OUTPUT_DIR}/order_effects_data.csv"
    output_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nSaved: {output_file}")
    print(f"Total rows: {len(output_df)}")
    print(f"Participants: {output_df['participant_id'].nunique()}")

    print("\nSummary:")
    print(output_df.groupby(['game_type', 'order_position'])['num_moves'].agg(['count', 'mean', 'std']))

    print("\nThis file is ready for R analysis!")
    print("Run R_data_analysis.R and it will include order effects analysis.")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import zipfile
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Output directory for saved files (Downloads - not in git repo)
output_dir = "/Users/rachelpapirmeister/Downloads"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
stats_file = f"{output_dir}/game_stats_{timestamp}.txt"

# Helper function to print and save to file
def log(text=""):
    print(text)
    stats_output.write(str(text) + "\n")

# Open stats file for writing
stats_output = open(stats_file, 'w')
log(f"Game Study Data Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

def read_json_from_zip(zip_path):
    results = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for file_name in zip_file.namelist():
            if file_name.endswith('.json'):
                with zip_file.open(file_name) as json_file:
                    content = json_file.read()
                    results[file_name] = json.loads(content)
    return results

data_file = read_json_from_zip('/Users/rachelpapirmeister/Downloads/DATA-20260107T001152Z-1-001.zip')

print("Files in the zip:")
for filename in data_file.keys():
    print(filename)

game_states = {}
eye_tracking = {}

for filename, data in data_file.items():
    if 'eye' in filename.lower():
        eye_tracking[filename] = data
    elif 'state' in filename.lower() or 'session' in filename.lower():
        game_states[filename] = data

print("Gamestates files:", len(game_states))
print("Eye-tracking files:", len(eye_tracking))


if game_states:
    first_gamestate_file = list(game_states.keys())[0]
    first_data = game_states[first_gamestate_file]
    print(f"\nFirst gamestate file: {first_gamestate_file}")
    print("Keys in the JSON:", list(first_data.keys()))

    # Convert the 'movements' list to a DataFrame (if it exists)
    if 'movements' in first_data and isinstance(first_data['movements'], list):
        df_movements = pd.DataFrame(first_data['movements'])
        print("\nMovements data:")
        print(df_movements.describe())

if eye_tracking:
    first_eye_file = list(eye_tracking.keys())[0]
    first_eye_data = eye_tracking[first_eye_file]
    print(f"\nFirst eye-tracking file: {first_eye_file}")
    print("Keys in the JSON:", list(first_eye_data.keys()))

    if 'gaze' in first_eye_data and isinstance(first_eye_data['gaze'], list):
        df_gaze = pd.DataFrame(first_eye_data['gaze'])
        print("\nGaze data:")
        print(df_gaze.describe())

game_summary = []

for filename, data in game_states.items():
    if 'Game A' in filename or 'game1' in filename:
        game_type = 'Game A'
    elif 'Game B' in filename or 'game2' in filename:
        game_type = 'Game B'
    elif 'Game C' in filename or 'game3' in filename:
        game_type = 'Game C'
    else:
        game_type = 'Unknown'


    if 'movements' in data:
        num_movements = len(data['movements'])
    elif 'events' in data:
        num_movements = len(data['events'])
    else:
        num_movements = 0

    game_summary.append({
        'filename': filename,
        'game_type': game_type,
        'num_movements': num_movements
    })

df_summary = pd.DataFrame(game_summary)

log("\n" + "="*50)
log("STATS ACROSS ALL GAMES")
log("="*50)

log(" ")
log("\nAverage movements per game:")
log(df_summary['num_movements'].mean())

log(" ")
log("\nAverage movements by game type:")
log(df_summary.groupby('game_type')['num_movements'].mean())

log(" ")
log("\nMovement statistics (count, mean, std, min, 25%, 50%, 75%, max):")
log(df_summary['num_movements'].describe())

log(" ")
log("\nStats by game type:")
log(df_summary.groupby('game_type')['num_movements'].describe())


gaze_summary = []

for filename, data in eye_tracking.items():
    if 'Game A' in filename or 'game1' in filename:
        game_type = 'Game A'
    elif 'Game B' in filename or 'game2' in filename:
        game_type = 'Game B'
    elif 'Game C' in filename or 'game3' in filename:
        game_type = 'Game C'
    else:
        game_type = 'Unknown'

    num_gaze = len(data.get('gaze', []))

    gaze_summary.append({
        'filename': filename,
        'game_type': game_type,
        'num_gaze': num_gaze
    })

df_gaze_summary = pd.DataFrame(gaze_summary)

log("\n" + "="*50)
log("GAZE STATS ACROSS ALL GAMES")
log("="*50)

log(" ")
log("\nAverage gaze points per session:")
log(df_gaze_summary['num_gaze'].mean())

log(" ")
log("\nAverage gaze points by game type:")
log(df_gaze_summary.groupby('game_type')['num_gaze'].mean())

log(" ")
log("\nGaze point statistics (count, mean, std, min, 25%, 50%, 75%, max):")
log(df_gaze_summary['num_gaze'].describe())

log(" ")
log("\nGaze stats by game type:")
log(df_gaze_summary.groupby('game_type')['num_gaze'].describe())


# histrogram:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

avg_movements = df_summary.groupby('game_type')['num_movements'].mean()
ax1.bar(avg_movements.index, avg_movements.values, color=['#3498db', '#e74c3c', '#2ecc71'])
ax1.set_title('Average Movements by Game Type')
ax1.set_xlabel('Game Type')
ax1.set_ylabel('Average Movements')

avg_gaze = df_gaze_summary.groupby('game_type')['num_gaze'].mean()
ax2.bar(avg_gaze.index, avg_gaze.values, color=['#3498db', '#e74c3c', '#2ecc71'])
ax2.set_title('Average Gaze Points by Game Type')
ax2.set_xlabel('Game Type')
ax2.set_ylabel('Average Gaze Points')

plt.tight_layout()
plt.savefig(f"{output_dir}/chart1_avg_movements_gaze_{timestamp}.png", dpi=150)
plt.show()


# K-means clustering:

log("\n" + "="*50)
log("K-MEANS CLUSTERING ON MOVEMENTS")
log("="*50)

all_movements = []

for filename, data in game_states.items():
    if 'movements' in data:

        if 'Game A' in filename or 'game1' in filename:
            game_type = 'Game A'
        elif 'Game B' in filename or 'game2' in filename:
            game_type = 'Game B'
        else:
            game_type = 'Unknown'

        for movement in data['movements']:
            # Extract features from each movement
            pos = movement.get('positionBefore', {})
            all_movements.append({
                'x': pos.get('x', 0),
                'y': pos.get('y', 0),
                'level': movement.get('level', 0),
                'direction': movement.get('direction', ''),
                'game_type': game_type
            })

# Create DataFrame
df_all_movements = pd.DataFrame(all_movements)
log(f"\nTotal movements collected: {len(df_all_movements)}")

# Prepare features for clustering (x, y, level)
features = df_all_movements[['x', 'y', 'level']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means with 4 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_all_movements['cluster'] = kmeans.fit_predict(features_scaled)

# Show cluster distribution
log("\nCluster distribution:")
log(df_all_movements['cluster'].value_counts().sort_index())

# Show cluster centers (unscaled)
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
log("\nCluster centers (x, y, level):")
for i, center in enumerate(centers):
    log(f"  Cluster {i}: x={center[0]:.1f}, y={center[1]:.1f}, level={center[2]:.1f}")

# Show cluster breakdown by game type
log("\nClusters by game type:")
log(pd.crosstab(df_all_movements['game_type'], df_all_movements['cluster']))

# Visualize as HEATMAPS (better for grid data)
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Heatmap 1: All movements - count at each grid position
heatmap_all = np.zeros((6, 6))
for _, row in df_all_movements.iterrows():
    x, y = int(row['x']), int(row['y'])
    if 0 <= x < 6 and 0 <= y < 6:
        heatmap_all[y, x] += 1

im1 = axes[0].imshow(heatmap_all, cmap='hot', origin='lower')
axes[0].set_title('Movement Density (All Games)')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')
plt.colorbar(im1, ax=axes[0], label='Count')

# Heatmap 2: Game A movements
heatmap_a = np.zeros((6, 6))
for _, row in df_all_movements[df_all_movements['game_type'] == 'Game A'].iterrows():
    x, y = int(row['x']), int(row['y'])
    if 0 <= x < 6 and 0 <= y < 6:
        heatmap_a[y, x] += 1

im2 = axes[1].imshow(heatmap_a, cmap='Blues', origin='lower')
axes[1].set_title('Movement Density (Game A)')
axes[1].set_xlabel('X Position')
axes[1].set_ylabel('Y Position')
plt.colorbar(im2, ax=axes[1], label='Count')

# Heatmap 3: Game B movements
heatmap_b = np.zeros((6, 6))
for _, row in df_all_movements[df_all_movements['game_type'] == 'Game B'].iterrows():
    x, y = int(row['x']), int(row['y'])
    if 0 <= x < 6 and 0 <= y < 6:
        heatmap_b[y, x] += 1

im3 = axes[2].imshow(heatmap_b, cmap='Reds', origin='lower')
axes[2].set_title('Movement Density (Game B)')
axes[2].set_xlabel('X Position')
axes[2].set_ylabel('Y Position')
plt.colorbar(im3, ax=axes[2], label='Count')

plt.tight_layout()
plt.savefig(f"{output_dir}/chart2_movement_density_heatmaps_{timestamp}.png", dpi=150)
plt.show()

# Bar chart showing clusters by GAME TYPE (clearer!)
fig, ax = plt.subplots(figsize=(10, 5))
cluster_game = df_all_movements.groupby(['cluster', 'game_type']).size().unstack(fill_value=0)
cluster_game.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])  # Blue = Game A, Red = Game B
ax.set_title('Cluster Distribution by Game Type')
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of Movements')
ax.legend(title='Game Type')

# Add labels to explain each cluster
cluster_labels = {
    0: 'Upper-left,\nLevel 2',
    1: 'Lower-left,\nLevel 5',
    2: 'Right side,\nLevel 2-3',
    3: 'Bottom-left,\nLevel 2'
}
for i, label in cluster_labels.items():
    ax.annotate(label, xy=(i, 0), xytext=(i, -1500), fontsize=8, ha='center', color='gray')

plt.tight_layout()
plt.savefig(f"{output_dir}/chart3_cluster_by_game_type_{timestamp}.png", dpi=150)
plt.show()

log("\n" + "="*50)
log("MOVEMENT DENSITY: LEVEL 1 vs LEVEL 5")
log("="*50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Game A - Level 1
heatmap_a1 = np.zeros((6, 6))
subset = df_all_movements[(df_all_movements['game_type'] == 'Game A') & (df_all_movements['level'] == 1)]
for _, row in subset.iterrows():
    x, y = int(row['x']), int(row['y'])
    if 0 <= x < 6 and 0 <= y < 6:
        heatmap_a1[y, x] += 1
im1 = axes[0, 0].imshow(heatmap_a1, cmap='Blues', origin='lower')
axes[0, 0].set_title(f'Game A - Level 1 ({len(subset)} movements)')
axes[0, 0].set_xlabel('X Position')
axes[0, 0].set_ylabel('Y Position')
plt.colorbar(im1, ax=axes[0, 0], label='Count')

# Game A - Level 5
heatmap_a5 = np.zeros((6, 6))
subset = df_all_movements[(df_all_movements['game_type'] == 'Game A') & (df_all_movements['level'] == 5)]
for _, row in subset.iterrows():
    x, y = int(row['x']), int(row['y'])
    if 0 <= x < 6 and 0 <= y < 6:
        heatmap_a5[y, x] += 1
im2 = axes[0, 1].imshow(heatmap_a5, cmap='Blues', origin='lower')
axes[0, 1].set_title(f'Game A - Level 5 ({len(subset)} movements)')
axes[0, 1].set_xlabel('X Position')
axes[0, 1].set_ylabel('Y Position')
plt.colorbar(im2, ax=axes[0, 1], label='Count')

# Game B - Level 1
heatmap_b1 = np.zeros((6, 6))
subset = df_all_movements[(df_all_movements['game_type'] == 'Game B') & (df_all_movements['level'] == 1)]
for _, row in subset.iterrows():
    x, y = int(row['x']), int(row['y'])
    if 0 <= x < 6 and 0 <= y < 6:
        heatmap_b1[y, x] += 1
im3 = axes[1, 0].imshow(heatmap_b1, cmap='Reds', origin='lower')
axes[1, 0].set_title(f'Game B - Level 1 ({len(subset)} movements)')
axes[1, 0].set_xlabel('X Position')
axes[1, 0].set_ylabel('Y Position')
plt.colorbar(im3, ax=axes[1, 0], label='Count')

# Game B - Level 5
heatmap_b5 = np.zeros((6, 6))
subset = df_all_movements[(df_all_movements['game_type'] == 'Game B') & (df_all_movements['level'] == 5)]
for _, row in subset.iterrows():
    x, y = int(row['x']), int(row['y'])
    if 0 <= x < 6 and 0 <= y < 6:
        heatmap_b5[y, x] += 1
im4 = axes[1, 1].imshow(heatmap_b5, cmap='Reds', origin='lower')
axes[1, 1].set_title(f'Game B - Level 5 ({len(subset)} movements)')
axes[1, 1].set_xlabel('X Position')
axes[1, 1].set_ylabel('Y Position')
plt.colorbar(im4, ax=axes[1, 1], label='Count')

plt.suptitle('Movement Density: Level 1 vs Level 5', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/chart4_level1_vs_level5_{timestamp}.png", dpi=150)
plt.show()

# Close stats file and print summary
stats_output.close()
print(f"\n{'='*50}")
print("FILES SAVED TO DOWNLOADS:")
print(f"{'='*50}")
print(f"  Stats: game_stats_{timestamp}.txt")
print(f"  Chart 1: chart1_avg_movements_gaze_{timestamp}.png")
print(f"  Chart 2: chart2_movement_density_heatmaps_{timestamp}.png")
print(f"  Chart 3: chart3_cluster_by_game_type_{timestamp}.png")
print(f"  Chart 4: chart4_level1_vs_level5_{timestamp}.png")

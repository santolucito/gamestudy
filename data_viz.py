import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Output directory for saved files (Downloads - not in git repo)
output_dir = "/Users/rachelpapirmeister/Downloads"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load your final dataset (after you've linked speech to movements)
# This assumes you've already run the NLP classifier and created NLP_features_for_LTA.csv
# NOTE: Data files are stored in Downloads to keep them off GitHub
csv_path = "/Users/rachelpapirmeister/Downloads/NLP_features_for_LTA.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} speech-movement episodes")
print(f"Speech categories: {df['speech_category'].value_counts()}")

# Calculate prop_direction_changes if it doesn't exist
if 'prop_direction_changes' not in df.columns:
    if 'direction_changes' in df.columns and 'num_moves' in df.columns:
        df['prop_direction_changes'] = df['direction_changes'] / df['num_moves'].replace(0, np.nan)
        print("Note: Calculated 'prop_direction_changes' from direction_changes / num_moves")

# Set seaborn style for prettier plots
sns.set_style("whitegrid")
sns.set_palette("husl")

#############################################
# VISUALIZATION 1: BOXPLOTS - CORE FINDING
#############################################

print("\n" + "="*50)
print("Creating Visualization 1: Movement Features by Category")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Movement entropy by category
sns.boxplot(data=df, x='speech_category', y='movement_entropy', ax=axes[0,0])
axes[0,0].set_title('Movement Entropy by Speech Category', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('Entropy (bits)')
axes[0,0].set_xlabel('')

# Direction changes
sns.boxplot(data=df, x='speech_category', y='direction_changes', ax=axes[0,1])
axes[0,1].set_title('Direction Changes by Speech Category', fontsize=12, fontweight='bold')
axes[0,1].set_ylabel('Number of Changes')
axes[0,1].set_xlabel('')

# Repeated sequences
sns.boxplot(data=df, x='speech_category', y='repeated_sequences', ax=axes[0,2])
axes[0,2].set_title('Repeated Sequences by Speech Category', fontsize=12, fontweight='bold')
axes[0,2].set_ylabel('Count')
axes[0,2].set_xlabel('')

# Unique positions visited
sns.boxplot(data=df, x='speech_category', y='unique_positions', ax=axes[1,0])
axes[1,0].set_title('Unique Positions Visited', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('Count')
axes[1,0].set_xlabel('Speech Category')

# Number of moves
sns.boxplot(data=df, x='speech_category', y='num_moves', ax=axes[1,1])
axes[1,1].set_title('Number of Moves', fontsize=12, fontweight='bold')
axes[1,1].set_ylabel('Count')
axes[1,1].set_xlabel('Speech Category')

# Backtracking
sns.boxplot(data=df, x='speech_category', y='num_revisits', ax=axes[1,2])
axes[1,2].set_title('Position Revisits (Backtracking)', fontsize=12, fontweight='bold')
axes[1,2].set_ylabel('Count')
axes[1,2].set_xlabel('Speech Category')

plt.suptitle('Movement Features by Speech Category', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f"{output_dir}/viz1_boxplots_by_category_{timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: viz1_boxplots_by_category_{timestamp}.png")
plt.show()

#############################################
# VISUALIZATION 2: FEATURE PROFILES HEATMAP
#############################################

print("\n" + "="*50)
print("Creating Visualization 2: Feature Profiles Heatmap")
print("="*50)

# Calculate mean features for each category
category_means = df.groupby('speech_category')[
    ['movement_entropy', 'direction_changes', 'repeated_sequences',
     'unique_positions', 'num_revisits', 'num_moves']
].mean()

# Normalize to 0-1 scale for better visualization
scaler = MinMaxScaler()
category_means_scaled = pd.DataFrame(
    scaler.fit_transform(category_means.T).T,
    index=category_means.index,
    columns=category_means.columns
)

# Reorder to: exploratory, confirmatory, exploitative
order = ['exploratory', 'confirmatory', 'exploitative']
category_means_scaled = category_means_scaled.reindex([cat for cat in order if cat in category_means_scaled.index])

# Create heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(category_means_scaled, annot=True, fmt='.2f', cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Mean Value'},
            linewidths=1, linecolor='white')
plt.title('Movement Feature Profiles by Speech Category', fontsize=14, fontweight='bold')
plt.ylabel('Speech Category', fontsize=12)
plt.xlabel('Movement Features', fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/viz2_feature_profiles_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: viz2_feature_profiles_heatmap_{timestamp}.png")
plt.show()

#############################################
# VISUALIZATION 3: VIOLIN PLOTS
#############################################

print("\n" + "="*50)
print("Creating Visualization 3: Violin Plots")
print("="*50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Movement entropy
sns.violinplot(data=df, x='speech_category', y='movement_entropy', ax=axes[0])
axes[0].set_title('Movement Entropy Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Entropy (bits)')
axes[0].set_xlabel('Speech Category')

# Direction changes (proportion)
if 'prop_direction_changes' in df.columns:
    sns.violinplot(data=df, x='speech_category', y='prop_direction_changes', ax=axes[1])
    axes[1].set_title('Proportion of Direction Changes', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Proportion')
else:
    sns.violinplot(data=df, x='speech_category', y='direction_changes', ax=axes[1])
    axes[1].set_title('Direction Changes', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
axes[1].set_xlabel('Speech Category')

# Repeated sequences
sns.violinplot(data=df, x='speech_category', y='repeated_sequences', ax=axes[2])
axes[2].set_title('Repeated Movement Sequences', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Count')
axes[2].set_xlabel('Speech Category')

plt.suptitle('Distribution of Movement Features by Speech Category',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}/viz3_violin_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: viz3_violin_plots_{timestamp}.png")
plt.show()

#############################################
# VISUALIZATION 4: PARTICIPANT DISTRIBUTION
#############################################

print("\n" + "="*50)
print("Creating Visualization 4: Participant Category Distribution")
print("="*50)

# Calculate proportion of each category per participant
participant_categories = df.groupby(['participant_id', 'speech_category']).size().unstack(fill_value=0)
participant_categories = participant_categories.div(participant_categories.sum(axis=1), axis=0)

# Stacked bar chart
fig, ax = plt.subplots(figsize=(14, 6))
participant_categories.plot(kind='bar', stacked=True, ax=ax,
                           color=['#ff9999', '#66b3ff', '#99ff99'])
ax.set_title('Proportion of Speech Categories by Participant', fontsize=14, fontweight='bold')
ax.set_xlabel('Participant', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/viz4_participant_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: viz4_participant_distribution_{timestamp}.png")
plt.show()

#############################################
# STATISTICAL TESTS
#############################################

print("\n" + "="*50)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*50)

stats_test_file = f"{output_dir}/statistical_tests_{timestamp}.txt"
with open(stats_test_file, 'w') as f:
    f.write("STATISTICAL TESTS FOR SPEECH-MOVEMENT ANALYSIS\n")
    f.write("="*60 + "\n\n")

    # Features to test
    features = ['movement_entropy', 'direction_changes', 'repeated_sequences',
                'unique_positions', 'num_moves', 'num_revisits']

    for feature in features:
        f.write(f"\n{feature.upper().replace('_', ' ')}:\n")
        f.write("-" * 60 + "\n")

        # Get data for each category
        exploratory = df[df['speech_category'] == 'exploratory'][feature].dropna()
        confirmatory = df[df['speech_category'] == 'confirmatory'][feature].dropna()
        exploitative = df[df['speech_category'] == 'exploitative'][feature].dropna()

        # Check if we have all three categories
        if len(exploratory) == 0 or len(confirmatory) == 0 or len(exploitative) == 0:
            f.write("  Skipping - not all categories present\n")
            continue

        # One-way ANOVA
        f_stat, p_value = f_oneway(exploratory, confirmatory, exploitative)

        f.write(f"  One-way ANOVA:\n")
        f.write(f"    F-statistic: {f_stat:.4f}\n")
        f.write(f"    p-value: {p_value:.4f}\n")

        if p_value < 0.001:
            f.write(f"    Result: *** HIGHLY SIGNIFICANT (p < 0.001)\n")
        elif p_value < 0.01:
            f.write(f"    Result: ** SIGNIFICANT (p < 0.01)\n")
        elif p_value < 0.05:
            f.write(f"    Result: * SIGNIFICANT (p < 0.05)\n")
        else:
            f.write(f"    Result: NOT SIGNIFICANT (p >= 0.05)\n")

        # If significant, do post-hoc tests
        if p_value < 0.05:
            f.write(f"\n  Post-hoc (Tukey HSD) pairwise comparisons:\n")

            # Prepare data for Tukey
            data_for_tukey = pd.DataFrame({
                'value': list(exploratory) + list(confirmatory) + list(exploitative),
                'category': ['exploratory']*len(exploratory) +
                           ['confirmatory']*len(confirmatory) +
                           ['exploitative']*len(exploitative)
            })

            tukey_result = pairwise_tukeyhsd(data_for_tukey['value'],
                                            data_for_tukey['category'],
                                            alpha=0.05)

            f.write(str(tukey_result))
            f.write("\n")

        # Descriptive statistics
        f.write(f"\n  Descriptive statistics:\n")
        f.write(f"    Exploratory:   M={exploratory.mean():.3f}, SD={exploratory.std():.3f}, n={len(exploratory)}\n")
        f.write(f"    Confirmatory:  M={confirmatory.mean():.3f}, SD={confirmatory.std():.3f}, n={len(confirmatory)}\n")
        f.write(f"    Exploitative:  M={exploitative.mean():.3f}, SD={exploitative.std():.3f}, n={len(exploitative)}\n")
        f.write("\n")

print(f"Saved: statistical_tests_{timestamp}.txt")

#############################################
# SUMMARY STATISTICS
#############################################

print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

# Save summary stats to text file
stats_file = f"{output_dir}/summary_statistics_{timestamp}.txt"
with open(stats_file, 'w') as f:
    f.write("SPEECH-MOVEMENT ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")

    f.write(f"Total episodes: {len(df)}\n")
    f.write(f"Participants: {df['participant_id'].nunique()}\n\n")

    f.write("Speech Category Distribution:\n")
    f.write(str(df['speech_category'].value_counts()) + "\n\n")

    f.write("Movement Features by Category:\n")
    f.write("="*60 + "\n")

    for category in ['exploratory', 'confirmatory', 'exploitative']:
        if category in df['speech_category'].values:
            f.write(f"\n{category.upper()}:\n")
            subset = df[df['speech_category'] == category]
            f.write(f"  Movement entropy: {subset['movement_entropy'].mean():.3f} (+/-{subset['movement_entropy'].std():.3f})\n")
            f.write(f"  Direction changes: {subset['direction_changes'].mean():.1f} (+/-{subset['direction_changes'].std():.1f})\n")
            f.write(f"  Repeated sequences: {subset['repeated_sequences'].mean():.1f} (+/-{subset['repeated_sequences'].std():.1f})\n")
            f.write(f"  Unique positions: {subset['unique_positions'].mean():.1f} (+/-{subset['unique_positions'].std():.1f})\n")
            f.write(f"  Number of moves: {subset['num_moves'].mean():.1f} (+/-{subset['num_moves'].std():.1f})\n")

print(f"Saved: summary_statistics_{timestamp}.txt")

print("\n" + "="*50)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*50)
print(f"Check your Downloads folder for:")
print(f"  - viz1_boxplots_by_category_{timestamp}.png")
print(f"  - viz2_feature_profiles_heatmap_{timestamp}.png")
print(f"  - viz3_violin_plots_{timestamp}.png")
print(f"  - viz4_participant_distribution_{timestamp}.png")
print(f"  - statistical_tests_{timestamp}.txt")
print(f"  - summary_statistics_{timestamp}.txt")

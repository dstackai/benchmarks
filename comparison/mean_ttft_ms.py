import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Read the CSV file
df = pd.read_csv('qps_comparison.csv')

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Define column names
gpu_col = 'gpu'
qps_col = 'qps'
mean_ttft_ms_col = 'mean_ttft_ms'

# Verify required columns
required_columns = [gpu_col, qps_col, mean_ttft_ms_col]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns {missing_columns}")
    sys.exit(1)

# Convert qps to string for categorical plotting
df[qps_col] = df[qps_col].astype(str)

# Filter out QPS values where not all GPUs have data
valid_qps = df.groupby(qps_col)[mean_ttft_ms_col].count()
valid_qps = valid_qps[valid_qps == df[gpu_col].nunique()].index
filtered_df = df[df[qps_col].isin(valid_qps)]

# Set seaborn style
sns.set(style="whitegrid")

# Initialize figure
plt.figure(figsize=(12, 8))

# Create bar plot
bar_plot = sns.barplot(
    x=qps_col,
    y=mean_ttft_ms_col,
    hue=gpu_col,
    data=filtered_df,
    dodge=True,
    palette="Set2"
)

# Set title and labels
plt.title('Mean TTFT (ms) vs QPS', fontsize=18)
plt.xlabel('QPS', fontsize=14)
plt.ylabel('Mean TTFT (ms)', fontsize=14)
plt.legend(title='GPU-Model')

# Add labels on top of bars
for p in bar_plot.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        bar_plot.annotate(f'{height:.1f}',
                          (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='bottom',
                          fontsize=10, color='black', xytext=(0, 5),
                          textcoords='offset points')

# Optimize layout
plt.tight_layout()

# Save the plot
plt.savefig('images/mean_ttft_vs_qps_comparison_gpus.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

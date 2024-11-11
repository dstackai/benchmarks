import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Read the CSV file
df = pd.read_csv('throughput_output.csv')

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Define column names
input_len_col = 'input_len'
out_len_col = 'out_len'
batch_size_col = 'batch_size'
tokens_per_second_col = 'tokens_per_second'

# Verify required columns
required_columns = [input_len_col, out_len_col, batch_size_col, tokens_per_second_col]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns {missing_columns}")
    sys.exit(1)

# Filter data
filtered_df = df[
    (df[input_len_col] == 32784) &
    (df[out_len_col] == 2048)
]

if filtered_df.empty:
    print("No data matches the specified input_len and out_len.")
    sys.exit(1)

# Convert batch_size to string for categorical plotting
filtered_df[batch_size_col] = filtered_df[batch_size_col].astype(str)

# Set seaborn style
sns.set(style="whitegrid")

# Initialize figure
plt.figure(figsize=(12, 8))

# Create bar plot
bar_plot = sns.barplot(
    x=batch_size_col,
    y=tokens_per_second_col,
    data=filtered_df,
    palette="Set2"
)

# Set title and labels
plt.title('Tokens Per Second vs Batch Size (Input_len=32784, Output_len=2048)', fontsize=18)
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('Tokens Per Second', fontsize=14)

# Add labels on top of bars
for p in bar_plot.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        bar_plot.annotate(f'{height:.1f}',
                          (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='bottom',
                          fontsize=10, color='black', xytext=(0, 5),
                          textcoords='offset points')

# Adjust y-axis limits based on data
plt.ylim(0, filtered_df[tokens_per_second_col].max() * 1.15)

# Optimize layout
plt.tight_layout()

# Save the plot
plt.savefig('tokens_per_second_vs_batch_size.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Read the CSV file
df = pd.read_csv('latency_output.csv')

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Define column names
input_len_col = 'input_len'
output_len_col = 'output_len'
end_to_end_latency_col = 'end_to_end_latency'
tokens_per_second_col = 'tokens_per_second'

# Verify required columns
required_columns = [input_len_col, output_len_col, end_to_end_latency_col, tokens_per_second_col]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns {missing_columns}")
    sys.exit(1)

# Set seaborn style
sns.set(style="whitegrid")

# Initialize figure
plt.figure(figsize=(12, 8))

# Create line plot without dots
sns.lineplot(
    x=end_to_end_latency_col,
    y=tokens_per_second_col,
    data=df
)

# Set title and labels
plt.title('Tokens Per Second vs End-to-End Latency', fontsize=18)
plt.xlabel('End-to-End Latency', fontsize=14)
plt.ylabel('Tokens Per Second', fontsize=14)

# Optimize layout
plt.tight_layout()

# Save the plot
plt.savefig('tokens_per_second_vs_end_to_end_latency.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
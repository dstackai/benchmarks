import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Read the CSV file
df = pd.read_csv('qps_output.csv')

# Debugging: Print original column names
print("Original Column Names:")
print(df.columns.tolist())

# Normalize column names: strip whitespace and convert to lowercase
df.columns = df.columns.str.strip().str.lower()

# Debugging: Print normalized column names
print("\nNormalized Column Names:")
print(df.columns.tolist())

# Define the exact column names based on normalized names
# Update these if your column names use underscores or different separators
enable_chunked_prefill_col = 'enable-chunked-prefill'      # Example: 'enable_chunked_prefill'
max_num_batched_tokens_col = 'max-num-batched-tokens'    # Example: 'max_num_batched_tokens'
max_num_seqs_col = 'max-num-seqs'                        # Example: 'max_num_seqs'
max_seq_len_to_capture_col = 'max-seq-len-to-capture'   # Example: 'max_seq_len_to_capture'
num_scheduler_step_col = 'num-scheduler-step'            # Example: 'num_scheduler_step'

# Verify if the necessary columns exist
required_columns = [
    enable_chunked_prefill_col,
    max_num_batched_tokens_col,
    max_num_seqs_col,
    max_seq_len_to_capture_col,
    num_scheduler_step_col,
    'qps',
    'request_throughput'
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"\nError: The following required columns are missing in the CSV: {missing_columns}")
    sys.exit(1)

# Ensure 'enable-chunked-prefill' is boolean
# If it's stored as string 'True'/'False', convert it
if df[enable_chunked_prefill_col].dtype == object:
    df[enable_chunked_prefill_col] = df[enable_chunked_prefill_col].str.lower() == 'true'

# Define the filtering criteria
desired_qps = [16, 32, 1000]
desired_scheduler_steps = [0]  # Only num-scheduler-step = 0
desired_enable_prefill = [True, False]

# Filter the data based on the specified criteria
filtered_df = df[
    (df[num_scheduler_step_col].isin(desired_scheduler_steps)) &
    (df[enable_chunked_prefill_col].isin(desired_enable_prefill)) &
    (df[max_num_batched_tokens_col] == 512) &
    (df[max_num_seqs_col] == 512) &
    (df[max_seq_len_to_capture_col] == 8192) &
    (df['qps'].isin(desired_qps))
]

# Check if the filtered dataframe is not empty
if filtered_df.empty:
    print("No data matches the specified criteria and desired QPS values.")
    sys.exit(1)
else:
    # Optional: Drop rows with missing values in 'request_throughput'
    filtered_df = filtered_df.dropna(subset=['request_throughput'])

    # Aggregate data in case there are multiple entries for the same QPS and enable-chunked-prefill
    # For example, take the mean of 'request_throughput'
    aggregated_df = filtered_df.groupby(['qps', enable_chunked_prefill_col])['request_throughput'].mean().reset_index()

    # Convert 'qps' to string for categorical plotting
    aggregated_df['qps'] = aggregated_df['qps'].astype(str)

    # Convert 'enable-chunked-prefill' to string for better labeling in the plot
    aggregated_df[enable_chunked_prefill_col] = aggregated_df[enable_chunked_prefill_col].astype(str)

    # Set the order of QPS and enable-chunked-prefill
    qps_order = [str(qps) for qps in desired_qps]                # ['16', '32', '1000']
    enable_prefill_order = ['True', 'False']                     # ['True', 'False']

    # Set the seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Define a contrasting color palette
    # Using 'Set1' which has distinct colors, suitable for two categories
    contrasting_palette = sns.color_palette("Set1", n_colors=len(enable_prefill_order))

    # Initialize the matplotlib figure
    plt.figure(figsize=(14, 8))

    # Create a grouped bar plot with seaborn
    bar_plot = sns.barplot(
        x='qps',
        y='request_throughput',
        hue=enable_chunked_prefill_col,
        data=aggregated_df,
        order=qps_order,
        hue_order=enable_prefill_order,
        palette=contrasting_palette,
        dodge=True
    )

    # Set the title and labels with increased font sizes
    plt.title('Request Throughput (Requests per Second) vs QPS for Different Enable Chunked Prefill Settings', fontsize=18)
    plt.xlabel('QPS (Queries Per Second)', fontsize=14)
    plt.ylabel('Request Throughput', fontsize=14)

    # Customize the legend with a title and adjust font sizes
    plt.legend(title='Enable Chunked Prefill', title_fontsize=14, fontsize=12)

    # Add numerical labels on top of each bar, excluding bars with request_throughput == 0
    for p in bar_plot.patches:
        height = p.get_height()
        if pd.notnull(height) and height > 0:
            bar_plot.annotate(f'{height:.1f}',
                              (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='bottom',
                              fontsize=10, color='black', xytext=(0, 5),
                              textcoords='offset points')

    # Optional: Adjust y-axis limits to accommodate labels
    plt.ylim(0, aggregated_df['request_throughput'].max() * 1.15)

    # Optimize layout to prevent clipping of labels/titles
    plt.tight_layout()

    # Save the plot to a file with high resolution
    plt.savefig('request_throughput_vs_qps_enable_chunked_prefill_contrasting_colors.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()
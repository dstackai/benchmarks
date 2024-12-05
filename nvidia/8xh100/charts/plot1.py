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
    'mean_ttft_ms'
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
desired_scheduler_steps = [1, 5, 10, 15]

# Filter the data based on the specified criteria
filtered_df = df[
    (df[enable_chunked_prefill_col] == True) &
    (df[max_num_batched_tokens_col] == 512) &
    (df[max_num_seqs_col] == 512) &
    (df[max_seq_len_to_capture_col] == 8192) &
    (df[num_scheduler_step_col].isin(desired_scheduler_steps)) &
    (df['qps'].isin(desired_qps))
]

# Check if the filtered dataframe is not empty
if filtered_df.empty:
    print("No data matches the specified criteria and desired QPS values.")
    sys.exit(1)
else:
    # Optional: Drop rows with missing values in 'mean_ttft_ms'
    filtered_df = filtered_df.dropna(subset=['mean_ttft_ms'])

    # Aggregate data in case there are multiple entries for the same QPS and scheduler step
    # For example, take the mean of 'mean_ttft_ms'
    aggregated_df = filtered_df.groupby(['qps', num_scheduler_step_col])['mean_ttft_ms'].mean().reset_index()

    # Convert 'qps' to string for categorical plotting
    aggregated_df['qps'] = aggregated_df['qps'].astype(str)

    # Convert 'num-scheduler-step' to string for better labeling in the plot
    aggregated_df[num_scheduler_step_col] = aggregated_df[num_scheduler_step_col].astype(str)

    # Set the order of QPS and scheduler steps
    qps_order = [str(qps) for qps in desired_qps]                # ['16', '32', '1000']
    scheduler_steps_order = [str(step) for step in desired_scheduler_steps]  # ['0', '5', '10', '15']

    # Set the seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Define a contrasting color palette
    # Using 'Set2' which has distinct colors, or you can choose 'tab10' or 'Set1'
    contrasting_palette = sns.color_palette("Set2", n_colors=len(scheduler_steps_order))

    # Initialize the matplotlib figure
    plt.figure(figsize=(14, 8))

    # Create a grouped bar plot with seaborn
    bar_plot = sns.barplot(
        x='qps',
        y='mean_ttft_ms',
        hue=num_scheduler_step_col,
        data=aggregated_df,
        order=qps_order,
        hue_order=scheduler_steps_order,
        palette=contrasting_palette,
        dodge=True
    )

    # Set the title and labels with increased font sizes
    plt.title('Mean TTFT (ms) vs QPS for Different Scheduler Steps', fontsize=18)
    plt.xlabel('QPS (Queries Per Second)', fontsize=14)
    plt.ylabel('Mean TTFT (ms)', fontsize=14)

    # Customize the legend with a title and adjust font sizes
    plt.legend(title='Num Scheduler Step', title_fontsize=14, fontsize=12)

    # Add numerical labels on top of each bar, excluding bars with mean_ttft_ms == 0
    for p in bar_plot.patches:
        height = p.get_height()
        if pd.notnull(height) and height > 0:
            bar_plot.annotate(f'{height:.1f}',
                              (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='bottom',
                              fontsize=10, color='black', xytext=(0, 5),
                              textcoords='offset points')

    # Optional: Adjust y-axis limits to accommodate labels
    plt.ylim(0, aggregated_df['mean_ttft_ms'].max() * 1.15)

    # Optimize layout to prevent clipping of labels/titles
    plt.tight_layout()

    # Save the plot to a file with high resolution
    plt.savefig('mean_ttft_vs_qps_scheduler_steps_contrasting_colors.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

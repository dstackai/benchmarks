import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse

def main(input_file, output_file):
    df = pd.read_csv(input_file)
    # Read the CSV file
    # df = pd.read_csv('qps_comparison.csv')

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Define column names
    gpu_col = 'gpu'
    qps_col = 'qps'
    request_throughput_col = 'request_throughput'

    # Verify required columns
    required_columns = [gpu_col, qps_col, request_throughput_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns {missing_columns}")
        sys.exit(1)

    # Convert qps to string for categorical plotting
    df[qps_col] = df[qps_col].astype(str)

    # Filter out QPS values where not all GPUs have data
    valid_qps = df.groupby(qps_col)[request_throughput_col].count()
    valid_qps = valid_qps[valid_qps == df[gpu_col].nunique()].index
    filtered_df = df[df[qps_col].isin(valid_qps)]

    # Set seaborn style
    sns.set(style="whitegrid")

    # Initialize figure
    plt.figure(figsize=(12, 8))

    # Create bar plot
    bar_plot = sns.barplot(
        x=qps_col,
        y=request_throughput_col,
        hue=gpu_col,
        data=filtered_df,
        dodge=True,
        palette="Set2",
        width=0.6
    )

    # Set title and labels
    plt.title('Request Throughput(request/s) vs QPS', fontsize=18)
    plt.xlabel('QPS', fontsize=14)
    plt.ylabel('Request Throughput', fontsize=14)
    plt.legend(title='GPU-Model', loc='upper left', bbox_to_anchor=(1, 1))

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
    # plt.savefig('images/request_throughput_vs_qps_comparison_gpus.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # Display the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a bar plot from CSV input.")
    parser.add_argument('--input-file', required=True, help="Path to the input CSV file.")
    parser.add_argument('--output-file', required=True, help="Path to save the output plot image.")
    args = parser.parse_args()

    main(args.input_file, args.output_file)

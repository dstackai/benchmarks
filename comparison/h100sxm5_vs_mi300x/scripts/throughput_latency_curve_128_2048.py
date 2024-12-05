import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
def main(input_file, output_file):
    df = pd.read_csv(input_file)
    # Read the CSV file
    # df = pd.read_csv('latency_comparision_128_2048.csv')

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Define column names
    gpu_col = 'gpu'
    input_len_col = 'input_len'
    output_len_col = 'output_len'
    batch_size_col = 'batch_size'
    end_to_end_latency_col = 'end_to_end_latency'
    tokens_per_second_col = 'tokens_per_second'

    # Verify required columns
    required_columns = [gpu_col, end_to_end_latency_col, tokens_per_second_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns {missing_columns}")
        sys.exit(1)

    # Set seaborn style
    sns.set(style="whitegrid")

    # Initialize figure
    plt.figure(figsize=(12, 8))

    # Create line plot for each GPU
    sns.lineplot(
        x=end_to_end_latency_col,
        y=tokens_per_second_col,
        hue=gpu_col,
        data=df,
        marker=None
    )

    # Set title and labels
    plt.title('Throughput Vs Latency (input_len=128, output_len=2048)', fontsize=18)
    plt.xlabel('End-to-End Latency', fontsize=14)
    plt.ylabel('Tokens Per Second', fontsize=14)
    plt.legend(title='GPU-Model')

    # Optimize layout
    plt.tight_layout()

    # Save the plot
    # plt.savefig('images/tokens_per_second_vs_end_to_end_latency_comparison_gpus_128_2048.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # Display the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a bar plot from CSV input.")
    parser.add_argument('--input-file', required=True, help="Path to the input CSV file.")
    parser.add_argument('--output-file', required=True, help="Path to save the output plot image.")
    args = parser.parse_args()

    main(args.input_file, args.output_file)



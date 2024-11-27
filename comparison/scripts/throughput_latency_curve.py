import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import argparse

def main(input_file, output_file, include_cost, input_len, output_len):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Filter dataframe based on input_len and output_len
    if input_len and output_len:
        df = df[(df['input_len'] == input_len) & (df['output_len'] == output_len)]


    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.lineplot(
        x='end_to_end_latency',
        y='tokens_per_second',
        hue='gpu',
        data=df,
        marker='o'
    )

    texts = []
    if include_cost:
        for i, point in df.iterrows():
            # Create text annotations with smaller font size
            text_annotation = ax.text(point['end_to_end_latency'], point['tokens_per_second'],
                                      f"{point['cost_per1m_token']:.2f}",
                                      ha='center', va='bottom', color=ax.get_lines()[-1].get_color(), fontsize='x-small')
            texts.append(text_annotation)

        # Use adjust_text to dynamically adjust text positions
        adjust_text(texts, ax=ax, expand_points=(1.2, 1.5), force_points=0.5,
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        # Update the output file name to reflect the filtering
        output_file = output_file.replace('.png', f'_with_cost.png')

    plt.title(f'Throughput Vs Latency (input_len={input_len}, output_len={output_len})', fontsize=18)
    plt.xlabel('End-to-End Latency', fontsize=14)
    plt.ylabel('Tokens Per Second', fontsize=14)
    plt.legend(title='GPU-Model')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a line plot from CSV input.")
    parser.add_argument('--input-file', required=True, help="Path to the input CSV file.")
    parser.add_argument('--output-file', required=True, help="Path to save the output plot image.")
    parser.add_argument('--cost', action='store_true', help="Include cost annotations in the plot.")
    parser.add_argument('--input-len', type=int, help="Input length to filter the data.")
    parser.add_argument('--output-len', type=int, help="Output length to filter the data.")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.cost, args.input_len, args.output_len)
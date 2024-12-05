import argparse
import csv
import json
import os
import re
import sys


def parse_filename(filename):
    """
    Extract input_len, out_len, and batch_size from the filename.
    Expected format: throughput_test_ip{input_len}_op{out_len}_np{batch_size}.json
    """
    pattern = r'throughput_test_ip(\d+)_op(\d+)_np(\d+)\.json$'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")

    input_len = int(match.group(1))
    out_len = int(match.group(2))
    batch_size = int(match.group(3))

    return input_len, out_len, batch_size


def read_json(json_path):
    """
    Read and parse the JSON file.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise IOError(f"Error reading JSON file '{json_path}': {e}")


def write_csv(output_path, headers, row):
    """
    Write a row to the CSV file. If the file does not exist, create it and write headers.
    """
    file_exists = os.path.isfile(output_path)

    try:
        with open(output_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        raise IOError(f"Error writing to CSV file '{output_path}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert throughput benchmarking JSON results to CSV.")
    parser.add_argument('-f', '--file', required=True, help='Path to the JSON file.')

    args = parser.parse_args()
    json_file = args.file

    if not os.path.isfile(json_file):
        print(f"Error: File '{json_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        input_len, out_len, batch_size = parse_filename(os.path.basename(json_file))
    except ValueError as ve:
        print(f"Error: {ve}", file=sys.stderr)
        sys.exit(1)

    try:
        json_data = read_json(json_file)
    except IOError as ioe:
        print(f"Error: {ioe}", file=sys.stderr)
        sys.exit(1)

    # Define the CSV headers
    headers = [
        'input_len',
        'out_len',
        'batch_size',
        'elapsed_time',
        'num_requests',
        'total_num_tokens',
        'requests_per_second',
        'tokens_per_second'
    ]

    # Prepare the row to write
    row = {
        'input_len': input_len,
        'out_len': out_len,
        'batch_size': batch_size,
        'elapsed_time': json_data.get('elapsed_time', ''),
        'num_requests': json_data.get('num_requests', ''),
        'total_num_tokens': json_data.get('total_num_tokens', ''),
        'requests_per_second': json_data.get('requests_per_second', ''),
        'tokens_per_second': json_data.get('tokens_per_second', '')
    }

    output_csv = 'throughput_output.csv'

    try:
        write_csv(output_csv, headers, row)
        print(f"Successfully appended data to '{output_csv}'.")
    except IOError as ioe:
        print(f"Error: {ioe}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

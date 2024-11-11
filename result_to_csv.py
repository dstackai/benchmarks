#!/usr/bin/env python3

import json
import csv
import argparse
import os
import sys

def str2bool(v):
    """
    Convert a string to a boolean.

    Args:
        v (str): The string to convert.

    Returns:
        bool: The boolean value.

    Raises:
        argparse.ArgumentTypeError: If the string is not a valid boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (true/false).')

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing input and output filenames, and additional parameters.
    """
    parser = argparse.ArgumentParser(description='Convert a JSON object to CSV with specified fields and additional parameters.')

    # Existing arguments
    parser.add_argument('-f', '--file', required=True, help='Path to the input JSON file.')
    parser.add_argument('-o', '--output', default='output.csv', help='Path to the output CSV file (default: output.csv).')

    # New arguments
    parser.add_argument('--num-scheduler-step', type=int, required=True, help='Number of scheduler steps (integer).')
    parser.add_argument('--qps', type=int, required=True, help='Queries per second (integer).')
    parser.add_argument('--enable-chunked-prefill', type=str2bool, required=True, help='Enable chunked prefill (boolean).')
    parser.add_argument('--max-num-batched-tokens', type=int, required=True, help='Maximum number of batched tokens (integer).')
    parser.add_argument('--max-num-seqs', type=int, required=True, help='Maximum number of sequences (integer).')
    parser.add_argument('--max-seq-len-to-capture', type=int, required=True, help='Maximum sequence length to capture (integer).')
    parser.add_argument('--enable-prefix-caching', type=str2bool, required=True, help='Enable prefix caching (boolean).')

    return parser.parse_args()

def load_json(file_path):
    """
    Load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    with open(file_path, 'r', encoding='utf-8') as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON: {e.msg}", e.doc, e.pos)
    return data

def extract_fields(data, fields):
    """
    Extract specified fields from JSON data.

    Args:
        data (dict): The JSON data.
        fields (list): List of fields to extract.

    Returns:
        dict: Dictionary containing the extracted fields.
    """
    extracted = {}
    for field in fields:
        key = field
        value = data.get(key, None)
        # Remove trailing colon for CSV header if present
        csv_header = field.rstrip(':')
        extracted[csv_header] = value
    return extracted

def read_existing_headers(output_file):
    """
    Read headers from an existing CSV file.

    Args:
        output_file (str): Path to the output CSV file.

    Returns:
        list: List of header fields.
    """
    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader, None)
    return headers if headers else []

def write_headers(output_file, fieldnames):
    """
    Write headers to a CSV file.

    Args:
        output_file (str): Path to the output CSV file.
        fieldnames (list): List of header fields.
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

def update_csv_with_new_headers(output_file, new_headers):
    """
    Update existing CSV with new headers by adding empty columns.

    Args:
        output_file (str): Path to the output CSV file.
        new_headers (list): List of all required header fields.
    """
    existing_headers = read_existing_headers(output_file)
    missing_headers = [header for header in new_headers if header not in existing_headers]

    if missing_headers:
        # Read existing data
        with open(output_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Update headers
        updated_headers = existing_headers + missing_headers

        # Write back with updated headers and add empty fields for missing headers
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=updated_headers)
            writer.writeheader()
            for row in rows:
                for header in missing_headers:
                    row[header] = None  # or any default value you prefer
                writer.writerow(row)
        print(f"Added new headers {missing_headers} to '{output_file}'.")

def write_csv(output_file, fieldnames, rows):
    """
    Write or append rows to a CSV file.

    Args:
        output_file (str): Path to the output CSV file.
        fieldnames (list): List of CSV header fields.
        rows (list of dict): List of dictionaries containing row data.
    """
    if os.path.exists(output_file):
        existing_headers = read_existing_headers(output_file)
        # Check if all required headers are present
        if not all(field in existing_headers for field in fieldnames):
            update_csv_with_new_headers(output_file, fieldnames)
        mode = 'a'
        write_header = False
    else:
        mode = 'w'
        write_header = True

    try:
        with open(output_file, mode=mode, newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()
                print(f"Writing headers to '{output_file}'.")
            writer.writerows(rows)
        if write_header:
            print(f"Data successfully written to '{output_file}' with headers.")
        else:
            print(f"Data successfully appended to '{output_file}'.")
    except IOError as e:
        print(f"IO error while writing to CSV: {e}")
        sys.exit(1)

def main():
    # Define the JSON fields to extract
    json_fields = [
        "num_prompts",
        "request_rate",
        "burstiness",
        "max_concurrency",
        "duration",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "request_goodput:",  # Note the trailing colon
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p99_itl_ms"
    ]

    # Define the additional arguments to include as CSV columns
    additional_fields = {
        "num-scheduler-step": None,
        "qps": None,
        "enable-chunked-prefill": None,
        "max-num-batched-tokens": None,
        "max-num-seqs": None,
        "max-seq-len-to-capture": None,
        "enable-prefix-caching": None
    }

    # Parse command-line arguments
    args = parse_arguments()

    # Load JSON data
    try:
        data = load_json(args.file)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit(1)
    except json.JSONDecodeError as json_error:
        print(json_error)
        sys.exit(1)

    # Extract filename from input file path
    input_filename = os.path.basename(args.file)
    # Optionally, remove extension if desired
    input_filename_no_ext = os.path.splitext(input_filename)[0]

    # Check if data is a list (multiple JSON objects) or a single object
    if isinstance(data, list):
        data_list = data
    elif isinstance(data, dict):
        data_list = [data]
    else:
        print("Unsupported JSON structure. Expected a list or a single JSON object.")
        sys.exit(1)

    # Extract specified JSON fields for each JSON object
    json_rows = [extract_fields(item, json_fields) for item in data_list]

    # Populate additional fields from command-line arguments for each row
    additional_rows = [{
        "num-scheduler-step": args.num_scheduler_step,
        "qps": args.qps,
        "enable-chunked-prefill": args.enable_chunked_prefill,
        "max-num-batched-tokens": args.max_num_batched_tokens,
        "max-num-seqs": args.max_num_seqs,
        "max-seq-len-to-capture": args.max_seq_len_to_capture,
        "enable-prefix-caching": args.enable_prefix_caching
    } for _ in data_list]

    # Populate the "filename" column for each row
    filename_rows = [{
        "filename": input_filename_no_ext
    } for _ in data_list]

    # Combine JSON data, additional fields, and filename for each row
    combined_rows = [ {**json_row, **additional_row, **filename_row} for json_row, additional_row, filename_row in zip(json_rows, additional_rows, filename_rows) ]

    # Define CSV headers (order: JSON fields first, then additional fields, then filename)
    csv_headers = [field.rstrip(':') for field in json_fields] + list(additional_fields.keys()) + ["filename"]

    # Write to CSV
    write_csv(args.output, csv_headers, combined_rows)

if __name__ == "__main__":
    main()
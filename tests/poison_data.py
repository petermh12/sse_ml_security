import pandas as pd
import random
import os
import argparse

def poison_data(input_file, output_file, poisoning_rate=0.1):
    """
    Introduce synthetic poisoned data into a given CSV file.

    Args:
        input_file (str): Path to the clean input CSV file.
        output_file (str): Path to save the poisoned output CSV file.
        poisoning_rate (float): Fraction of rows to poison (default 10%).
    """
    print(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file)

    # Ensure there are text-based columns to poison
    text_columns = df.select_dtypes(include=['object']).columns
    if text_columns.empty:
        raise ValueError("No text columns found in the input file to poison.")

    # Poison specified fraction of rows
    num_rows_to_poison = int(len(df) * poisoning_rate)
    print(f"Poisoning {num_rows_to_poison} rows ({poisoning_rate*100}% of total rows).")

    poisoned_indices = random.sample(range(len(df)), num_rows_to_poison)
    poison_payloads = ["Credit Card: 193472198328364", "attack_payload", "SSN: 123-45-6789", "Birthday: 01/01/1970", "Full Name: John Doe", "Credit Card: 1234567890123456", "password: mypassword", "SSN: 987-65-4321", "Birthday: 12/31/1999", "Full Name: Jane Smith"]

    for idx in poisoned_indices:
        for col in text_columns:
            df.at[idx, col] = random.choice(poison_payloads)

    # Save poisoned file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Poisoned data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a clean CSV file for testing.")
    parser.add_argument("input_file", help="Path to the clean input CSV file.")
    parser.add_argument("output_file", help="Path to save the poisoned CSV file.")
    parser.add_argument("--rate", type=float, default=0.1, help="Poisoning rate (default: 0.1)")

    args = parser.parse_args()

    poison_data(args.input_file, args.output_file, args.rate)

import pandas as pd
import argparse
import os

def evaluate_cleaning(poisoned_file, cleaned_file, poison_keywords):
    """
    Evaluate the effectiveness of the data cleaning pipeline.

    Args:
        poisoned_file (str): Path to the poisoned CSV file.
        cleaned_file (str): Path to the cleaned CSV file.
        poison_keywords (list): List of keywords that indicate poisoned data.
    """
    print("\n--- Cleaning Effectiveness Report ---")

    # Load the data
    poisoned_df = pd.read_csv(poisoned_file)
    cleaned_df = pd.read_csv(cleaned_file)

    # Identify poisoned rows
    poisoned_indices = poisoned_df.index[
        poisoned_df.apply(lambda row: any(kw in str(row).lower() for kw in poison_keywords), axis=1)
    ]
    detected_indices = poisoned_indices[~poisoned_indices.isin(cleaned_df.index)]

    num_poisoned = len(poisoned_indices)
    num_detected = len(detected_indices)

    # Report results
    print(f"Total Poisoned Samples: {num_poisoned}")
    print(f"Detected and Removed: {num_detected}")
    detection_rate = (num_detected / num_poisoned) * 100 if num_poisoned > 0 else 0
    print(f"Detection Rate: {detection_rate:.2f}%")
    print("--------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate data cleaning effectiveness.")
    parser.add_argument("poisoned_file", help="Path to the poisoned CSV file.")
    parser.add_argument("cleaned_file", help="Path to the cleaned CSV file.")
    parser.add_argument("--keywords", nargs="+", default=["malicious_input", "attack_payload", "poisoned_data_sample"],
                        help="List of keywords representing poisoned data.")

    args = parser.parse_args()

    evaluate_cleaning(args.poisoned_file, args.cleaned_file, args.keywords)
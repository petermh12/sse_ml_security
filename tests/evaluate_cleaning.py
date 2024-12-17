import pandas as pd
import argparse

def evaluate_cleaning(poisoned_file, cleaned_file, poison_keywords):
    """
    Evaluate the effectiveness of the data cleaning pipeline by comparing content.

    Args:
        poisoned_file (str): Path to the poisoned CSV file.
        cleaned_file (str): Path to the cleaned CSV file.
        poison_keywords (list): List of keywords that indicate poisoned data.
    """
    print("\n--- Cleaning Effectiveness Report ---")

    # Load the data
    poisoned_df = pd.read_csv(poisoned_file)
    cleaned_df = pd.read_csv(cleaned_file)

    # Identify poisoned rows in the original data
    def is_poisoned(row):
        return any(kw in str(row).lower() for kw in poison_keywords)

    poisoned_rows = poisoned_df[poisoned_df.apply(is_poisoned, axis=1)]
    total_poisoned = len(poisoned_rows)

    # Check how many poisoned rows are left in the cleaned data
    remaining_poisoned_rows = cleaned_df[cleaned_df.apply(is_poisoned, axis=1)]
    num_remaining_poisoned = len(remaining_poisoned_rows)

    # Calculate removed rows
    num_removed_poisoned = total_poisoned - num_remaining_poisoned
    detection_rate = (num_removed_poisoned / total_poisoned) * 100 if total_poisoned > 0 else 0

    # Report results
    print(f"Total Poisoned Samples: {total_poisoned}")
    print(f"Detected and Removed: {num_removed_poisoned}")
    print(f"Remaining Poisoned Samples: {num_remaining_poisoned}")
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

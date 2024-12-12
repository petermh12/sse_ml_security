import pandas as pd
import os

def label_consistency_check(df, file, label_column='label'):
    print(f"\nChecking label consistency in {file}...")
    try:
        label_counts = df[label_column].value_counts()
        print(f"Label distribution:\n{label_counts}")
    except KeyError:
        print(f"Label column '{label_column}' not found in {file}.")
        return
    
    # Check for class imbalance
    imbalance_ratio = label_counts.min() / label_counts.max()
    if imbalance_ratio < 0.1:
        print("Warning: Unusual label distribution detected (possible label flipping)!")
    else:
        print("Label distribution appears normal.")

if __name__ == "__main__":
    print("\n\n---------------------Final Step: Label Validation-------------------------\n\n")
    for file in os.listdir("data/processed/post_autoencoder"):
        if file == "sample.csv":
            continue
        df = pd.read_csv(f"data/processed/post_autoencoder/{file}")
        labels=label_consistency_check(df, file)

    print("Label validation workflow completed successfully.\n\n")
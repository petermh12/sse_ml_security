import pandas as pd

def label_consistency_check(df, label_column='label'):
    print("Checking label consistency...")
    label_counts = df[label_column].value_counts()
    print(f"Label distribution:\n{label_counts}")
    
    # Check for class imbalance
    imbalance_ratio = label_counts.min() / label_counts.max()
    if imbalance_ratio < 0.1:
        print("Warning: Unusual label distribution detected (possible label flipping)!")
    else:
        print("Label distribution appears normal.")

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/cleaned_training_data.csv")
    label_consistency_check(df)


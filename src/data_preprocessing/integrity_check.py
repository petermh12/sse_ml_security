import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def data_integrity_check(df):
    print("Checking data integrity...")
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Dropping them.")
        df = df.dropna()
    else:
        print("No missing values found.")
    
    # Additional integrity checks (e.g., removing duplicates)
    df.drop_duplicates(inplace=True)
    return df

if __name__ == "__main__":
    df = load_data("../../data/raw/training_data.csv")
    df_clean = data_integrity_check(df)
    df_clean.to_csv("../../data/processed/cleaned_data.csv", index=False)
    print("Cleaned data saved to '../../data/processed/cleaned_data.csv'.")


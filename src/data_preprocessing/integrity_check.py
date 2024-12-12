import pandas as pd
import os

def load_data(filepath):
    # Load data from a CSV file
    print(f"Reading {filepath} data")
    if os.path.isfile(filepath) and filepath.endswith(".csv"):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"\nError loading data from {filepath}: {e}")
            if 'Error tokenizing data' in str(e):
                print(f"Changing file to csv from scsv\n")
                df = pd.read_csv(filepath, sep=';')
        print(f"Data loaded from {filepath}")
        # Append the dataframe only if it is successfully loaded
        print(df.shape)
    else:
        print(f"Skipped {filepath}, not a valid file.")

    #add index column if not present
    if df.columns[0] != 'index':    
        df.insert(0,'index',range(1, len(df) + 1))
    return df

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
    print("\n\n---------------------First Step: Integrity Check-------------------------\n\n")
    for file in os.listdir("data/raw"):
        if file == "sample.csv":
            continue
        df = load_data(f"data/raw/{file}")
        if df.empty:
            print("No data loaded. Exiting...")
        else:
            df_clean = data_integrity_check(df)
            df_clean.to_csv(f"data/processed/cleaned/cleaned_{file}", index=False)
            rows, columns = df_clean.shape
            print(f"Cleaned data saved to 'data/processed/cleaned/cleaned_data.csv'.\nRows: {rows}, Columns: {columns}")
    print("Data screening workflow completed successfully.\n\n")

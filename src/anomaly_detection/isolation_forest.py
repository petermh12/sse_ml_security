import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

def detect_anomalies(df):
    print("Detecting anomalies using Isolation Forest...")
    
    # Select only numeric columns
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.empty:
        print("No numeric columns found. Skipping anomaly detection.")
        return df, pd.DataFrame()  # Return original DataFrame and an empty anomalies DataFrame

    # Scale numeric data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_data)
    
    # Fit Isolation Forest
    isolation_forest = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = isolation_forest.fit_predict(df_scaled)
    
    # Separate anomalies
    anomalies = df[df['anomaly'] == -1]
    print(f"Detected {len(anomalies)} anomalies.")
    
    # Return clean data (non-anomalies) and anomalies
    return df[df['anomaly'] == 1].drop(columns=['anomaly']), anomalies

if __name__ == "__main__":
    print("\n\n---------------------Next Step: Iso Forest-------------------------\n\n")
    for file in os.listdir("data/processed/cleaned"):
        filepath = f"data/processed/cleaned/{file}"
        if os.path.isfile(filepath):
            print(f"Processing file: {filepath}")
            try:
                df = pd.read_csv(filepath)
                df_filtered, anomalies = detect_anomalies(df)
                
                # Save results
                df_filtered.to_csv(f"data/processed/training/training_{file}", index=False)
                anomalies.to_csv(f"data/processed/anomalies/anomalies_{file}", index=False)
                print(f"Filtered data saved to 'data/processed/training/training_{file}'.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    print("Anomaly detection workflow completed successfully.\n\n")

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(df):
    print("Detecting anomalies using Isolation Forest...")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    isolation_forest = IsolationForest(contamination=0.02, random_state=42)  # 2% expected anomalies
    df['anomaly'] = isolation_forest.fit_predict(df_scaled)
    
    anomalies = df[df['anomaly'] == -1]
    print(f"Detected {len(anomalies)} anomalies.")
    return df[df['anomaly'] == 1], anomalies  # Return clean data and anomalies

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/cleaned_data.csv")
    df_filtered, anomalies = detect_anomalies(df)
    
    df_filtered.to_csv("../../data/processed/cleaned_training_data.csv", index=False)
    anomalies.to_csv("../../data/processed/anomalies.csv", index=False)
    print("Filtered data saved to '../../data/processed/cleaned_training_data.csv'.")


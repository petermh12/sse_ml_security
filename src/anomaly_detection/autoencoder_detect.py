import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath).select_dtypes(include=[np.number])
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(data):
    autoencoder = build_autoencoder(data.shape[1])
    autoencoder.fit(data, data, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    reconstructions = autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    
    threshold = np.percentile(mse, 95)  # Set threshold at 95th percentile
    anomalies = mse > threshold
    
    print(f"Detected {np.sum(anomalies)} potential adversarial samples.")
    return anomalies

if __name__ == "__main__":
    data = load_and_preprocess("../../data/processed/cleaned_training_data.csv")
    anomalies = train_autoencoder(data)
    
    df = pd.read_csv("../../data/processed/cleaned_training_data.csv")
    df_filtered = df[~anomalies]
    
    df_filtered.to_csv("../../data/processed/cleaned_autoencoder_data.csv", index=False)
    print("Cleaned data saved to '../../data/processed/cleaned_autoencoder_data.csv'.")


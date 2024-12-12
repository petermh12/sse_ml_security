import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import _KerasLazyLoader
from keras import models
from keras import layers
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(16, activation='relu')(input_layer)
    encoded = layers.Dense(8, activation='relu')(encoded)
    decoded = layers.Dense(16, activation='relu')(encoded)
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(data):
    autoencoder = build_autoencoder(data.shape[1])
    autoencoder.fit(data, data, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    
    reconstructions = autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    
    threshold = np.percentile(mse, 95)  # Set threshold at 95th percentile
    anomalies = mse > threshold
    
    print(f"Detected {np.sum(anomalies)} potential adversarial samples.")
    return anomalies

if __name__ == "__main__":

    print("\n\n---------------------Next Step: Autoencoder-------------------------\n\n")
    for file in os.listdir("data/processed/training"):
        data = load_and_preprocess(f"data/processed/training/{file}")
        print(f"\nProcessing file: {file}\n")
        anomalies = train_autoencoder(data)
    
        df = pd.read_csv(f"data/processed/training/{file}")
        df_filtered = df[~anomalies]
        file_name= file.split("_")[2]
        df_filtered.to_csv(f"data/processed/post_autoencoder/cleaned_autoencoder_{file_name}", index=False)
        print(f"Cleaned data saved to 'data/processed/post_autoencoder/cleaned_autoencoder_{file_name}'.")

    print("Autoencoder detection workflow completed successfully.\n\n")
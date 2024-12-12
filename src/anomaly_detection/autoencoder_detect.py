import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import _KerasLazyLoader
from keras import models
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def load_and_preprocess(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Identify text columns
    text_columns = data.select_dtypes(include=['object']).columns

    # Encode each text column with TF-IDF and collect the results
    tfidf_encodings = []
    for col in text_columns:
        vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
        tfidf_matrix = vectorizer.fit_transform(data[col].fillna('')).toarray()  # Handle NaN
        tfidf_encodings.append(tfidf_matrix)
    
    # Concatenate all TF-IDF encoded columns
    if tfidf_encodings:
        combined_tfidf = np.hstack(tfidf_encodings)
    else:
        combined_tfidf = np.empty((len(data), 0))  # No text columns, empty array
    
    # Handle numerical columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    numeric_data = data[numeric_columns].fillna(0).to_numpy()  # Fill NaNs with 0
    
    # Combine numeric and TF-IDF encoded features
    combined_data = np.hstack([numeric_data, combined_tfidf])
    
    # Scale the combined data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    return scaled_data, text_columns  # Return text columns for reconstruction if needed


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
        data, text_cols = load_and_preprocess(f"data/processed/training/{file}")
        print(f"\nProcessing file: {file}\n")
        anomalies = train_autoencoder(data)
    
        df = pd.read_csv(f"data/processed/training/{file}")
        df_filtered = df[~anomalies]
        file_name= file.split("_")[2]
        df_filtered.to_csv(f"data/processed/post_autoencoder/cleaned_autoencoder_{file_name}", index=False)
        print(f"Cleaned data saved to 'data/processed/post_autoencoder/cleaned_autoencoder_{file_name}'.")

    print("Autoencoder detection workflow completed successfully.\n\n")
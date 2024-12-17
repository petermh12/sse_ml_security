# Secure ML Data Screening

This project focuses on securing training data for ML systems against data poisoning and adversarial attacks.

## Project Structure

- **data/**: Contains raw and processed datasets.
- **src/**: Core source code for data preprocessing, anomaly detection, and validation.
- **notebooks/**: Jupyter notebooks for analysis.
- **tests/**: Unit tests for each module.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. To run from uppermost directory level:
   python3 src/main.py


## Testing

1. **Run the following:** python tests/poison_data.py data/raw/my_clean_data.csv data/processed/training/poisoned_data.csv --rate 0.2
   - This will poison the input file to a given percentage

2. Run the main.py script with the poisoned data, with the results stored in data/processed/post_autoencoder

3. **Run the following:** python tests/evaluate_cleaning.py data/processed/training/poisoned_data.csv data/processed/post_autoencoder/cleaned_autoencoder_data.csv
   - This will test the poisoned file against the newly cleaned one to learn the effectiveness


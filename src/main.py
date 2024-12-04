import subprocess

def run_module(module):
    result = subprocess.run(["python", module], check=True)
    if result.returncode == 0:
        print(f"{module} executed successfully.")

if __name__ == "__main__":
    modules = [
        "./data_preprocessing/integrity_check.py",
        "./anomaly_detection/isolation_forest.py",
        "./anomaly_detection/autoencoder_detect.py",
        "./data_preprocessing/label_validation.py"
    ]

    for module in modules:
        run_module(module)
    print("Data screening workflow completed successfully.")


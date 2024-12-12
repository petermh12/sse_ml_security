import subprocess

def run_module(module):
    result = subprocess.run(["python3", module], check=True)
    if result.returncode == 0:
        print(f"{module} executed successfully.")

if __name__ == "__main__":
    modules = [
        "src/data_preprocessing/integrity_check.py",
        "src/anomaly_detection/isolation_forest.py",
        "src/anomaly_detection/autoencoder_detect.py",
        "src/data_preprocessing/label_validation.py"
    ]

    for module in modules:
        run_module(module)
    print("\n--------------------Data screening workflow completed successfully.----------------------\n")


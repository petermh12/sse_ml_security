import unittest
import pandas as pd
from src.anomaly_detection.isolation_forest import detect_anomalies

class TestAnomalyDetection(unittest.TestCase):
    def test_detect_anomalies(self):
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 100],  # 100 is an outlier
            'feature2': [4, 5, 6, 7]
        })
        clean_data, anomalies = detect_anomalies(data)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies.iloc[0]['feature1'], 100)

if __name__ == '__main__':
    unittest.main()


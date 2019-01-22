
from external.luminol.src import luminol
from external.luminol.src.luminol.anomaly_detector import AnomalyDetector

from data.load import load

data = load().ekg()[0:1000]




detector = AnomalyDetector(data)
anomalies = detector.get_anomalies()

print(anomalies)


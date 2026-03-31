# detection_agent.py
# Runs the trained Isolation Forest model to score each per-user-hour feature row.

import joblib
import pandas as pd

# Must stay in sync with analysis_agent.py FEATURE_COLS and models/train_model.py
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_after_hours",
    "logon_count",
    "logoff_count",
    "usb_connect",
    "usb_disconnect",
    "file_count",
    "http_count",
    "email_count",
    "email_size_total",
    "email_attachments_total",
]


class DetectionAgent:
    def __init__(self, model_path: str = "models/isolation_forest.pkl"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = joblib.load(self.model_path)
        print(f"[DetectionAgent] Model loaded from {self.model_path}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        X = df[FEATURE_COLS].fillna(0)
        df = df.copy()
        df["anomaly_score"] = self.model.decision_function(X)
        df["is_anomaly"]    = self.model.predict(X) == -1
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.load_model()
        result = self.predict(df)
        anomalies = int(result["is_anomaly"].sum())
        print(f"[DetectionAgent] {anomalies:,} anomalies detected out of {len(result):,} records.")
        return result


if __name__ == "__main__":
    print("[DetectionAgent] Run via pipeline.py")

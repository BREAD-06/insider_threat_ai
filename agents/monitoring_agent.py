# monitoring_agent.py
# Loads all 5 CERT r4.2 CSV files from a directory and returns a single merged DataFrame.
# Uses column selection and chunked reading to handle very large files (http.csv is ~14 GB).

import os
import pandas as pd


# Only load the columns we actually need from each file.
# Skipping 'content', 'filename', 'url' (free-text / large fields we don't use for features).
CERT_FILES = {
    "logon":  {"file": "logon.csv",  "cols": ["date", "user", "pc", "activity"]},
    "device": {"file": "device.csv", "cols": ["date", "user", "pc", "activity"]},
    "file":   {"file": "file.csv",   "cols": ["date", "user", "pc"]},
    "http":   {"file": "http.csv",   "cols": ["date", "user", "pc"]},
    "email":  {"file": "email.csv",  "cols": ["date", "user", "pc", "size", "attachments"]},
}

# Rows per chunk when reading large CSVs
CHUNK_SIZE = 200_000


class MonitoringAgent:
    def __init__(self, data_dir: str = "data/cert_r4.2"):
        self.data_dir = data_dir

    def _load_file(self, name: str, spec: dict) -> pd.DataFrame:
        path = os.path.join(self.data_dir, spec["file"])
        if not os.path.exists(path):
            print(f"[MonitoringAgent] WARNING: {path} not found — skipping.")
            return pd.DataFrame()

        cols = spec["cols"]
        frames = []
        try:
            for chunk in pd.read_csv(
                path,
                usecols=cols,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):
                frames.append(chunk)
        except Exception as e:
            print(f"[MonitoringAgent] ERROR reading {spec['file']}: {e}")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["log_type"] = name
        return df

    def collect_logs(self) -> pd.DataFrame:
        """Load and concatenate selected columns from all CERT CSV files."""
        frames = []
        for name, spec in CERT_FILES.items():
            df = self._load_file(name, spec)
            if not df.empty:
                frames.append(df)
                print(f"[MonitoringAgent]  {spec['file']:<12} → {len(df):>10,} rows")

        if not frames:
            raise FileNotFoundError(
                f"No CERT CSV files found in '{self.data_dir}'. "
                "Please extract the dataset there."
            )
        return pd.concat(frames, ignore_index=True, sort=False)

    def run(self) -> pd.DataFrame:
        print(f"[MonitoringAgent] Loading CERT r4.2 logs from '{self.data_dir}' …")
        df = self.collect_logs()
        print(f"[MonitoringAgent] Total raw records loaded: {len(df):,}")
        return df


if __name__ == "__main__":
    agent = MonitoringAgent()
    df = agent.run()
    print(df["log_type"].value_counts())

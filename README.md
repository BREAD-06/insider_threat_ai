# 🛡️ Insider Threat AI

An AI-powered multi-agent pipeline for detecting insider threats using the **CERT Insider Threat Dataset (Release 4.2)**. The system uses an **Isolation Forest** anomaly detection model combined with rule-based verification agents to identify suspicious employee behaviour across login, USB, file, web, and email activity logs.

---

## 🏗️ Architecture

```
Monitor → Analyse → Detect → Verify → Respond → Learn
```

| Agent | Role |
|-------|------|
| `MonitoringAgent` | Loads all 5 CERT r4.2 CSVs using chunked reading |
| `AnalysisAgent` | Engineers 12 per-user-hour features from raw logs |
| `DetectionAgent` | Scores each row with a trained Isolation Forest model |
| `VerificationAgent` | Cross-checks anomalies with 6 rule-based heuristics |
| `ResponseAgent` | Logs confirmed threats to `data/alerts.jsonl` |
| `LearningAgent` | Retrains the model on demand |

---

## 📁 Dataset

This project uses the **CERT Insider Threat Dataset (Release 4.2)** — a benchmark dataset from Carnegie Mellon University's Software Engineering Institute.

**Download:** https://www.kaggle.com/datasets/andrihjonior/cert-insider-threat-dataset-r4-2

After downloading, extract the 5 CSV files into:

```
data/
└── cert_r4.2/
    ├── logon.csv       (~855k rows)
    ├── device.csv      (~405k rows)
    ├── file.csv        (~446k rows)
    ├── http.csv        (~28.4M rows, 13.8 GB)
    └── email.csv       (~2.6M rows)
```

> ⚠️ The dataset files are NOT included in this repo due to their size.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python models/train_model.py
```
This reads all 5 CERT CSVs, engineers features, and saves `models/isolation_forest.pkl`.  
*(Takes ~15–20 min depending on hardware due to the 28M-row http.csv)*

### 3. Run the full pipeline
```bash
python pipeline.py
```

### 4. Launch the dashboard
```bash
python -m streamlit run dashboard/app.py --browser.gatherUsageStats false
```
Then open **http://localhost:8501** in your browser.

---

## 📊 Features Engineered

From the 5 raw CERT log files, the `AnalysisAgent` builds a **per-user × per-hour** feature table:

| Feature | Source |
|---------|--------|
| `hour` | All logs |
| `day_of_week` | All logs |
| `is_after_hours` | Derived (hour < 7 or > 20) |
| `logon_count` | `logon.csv` |
| `logoff_count` | `logon.csv` |
| `usb_connect` | `device.csv` |
| `usb_disconnect` | `device.csv` |
| `file_count` | `file.csv` |
| `http_count` | `http.csv` |
| `email_count` | `email.csv` |
| `email_size_total` | `email.csv` |
| `email_attachments_total` | `email.csv` |

---

## 🔍 Verification Rules

Anomalies are confirmed threats if **any** rule fires:

| Rule | Condition |
|------|-----------|
| After-hours activity | Hour < 7 or > 20 |
| USB device connected | `usb_connect > 0` |
| High file volume | `file_count > 50` |
| Mass email | `email_count > 20` |
| Large email size | `email_size_total > 5 MB` |
| Excessive browsing | `http_count > 100` |

---

## 📂 Project Structure

```
insider_threat_ai/
├── agents/
│   ├── monitoring_agent.py     # Load 5 CERT CSVs
│   ├── analysis_agent.py       # Feature engineering
│   ├── detection_agent.py      # Isolation Forest scoring
│   ├── verification_agent.py   # Rule-based verification
│   ├── response_agent.py       # Alert logging
│   └── learning_agent.py       # Model retraining
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── models/
│   └── train_model.py          # Training script
├── api/
│   └── main.py                 # FastAPI REST API
├── data/
│   └── cert_r4.2/              # ← Place dataset here (not in repo)
├── pipeline.py                 # End-to-end orchestrator
└── requirements.txt
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — Isolation Forest
- **pandas** — data processing (chunked reading for large files)
- **Streamlit** — dashboard
- **Plotly** — interactive charts
- **FastAPI** — REST API
- **joblib** — model persistence

---

## 📜 License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Carnegie Mellon University SEI  
Code: MIT

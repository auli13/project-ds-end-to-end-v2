# 🏭 Factory Predictive Maintenance

> Predict a Mechanical/Electrical failure 30 minutes before it happens — across an 8-station factory line — and explain why, live, through a GenAI-powered dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-scikit--learn-orange?style=flat)
![Gemini](https://img.shields.io/badge/Google_Gemini-3.7_Flash-4285F4?style=flat&logo=google&logoColor=white)

**Live app:** `<PENDING — add your deployed Streamlit Community Cloud URL here>`

---

## 📌 Overview

Unplanned downtime is expensive, and often preventable. This project simulates an 8-station factory line, trains a model per station to catch Mechanical/Electrical failures ahead of time, and wraps it all in an interactive dashboard with a Gemini-powered assistant that explains *why*, not just *what*.

1. **Simulate** a realistic 8-station factory line with physics-based failures
2. **Engineer** causal, leak-free features from sensor readings
3. **Train & calibrate** Random Forest / XGBoost per station, tuned to a station-specific Recall floor
4. **Deploy** through a Streamlit dashboard with an 8-tool Gemini agent

---

## 🔁 Pipeline

```
Factory Simulator → Clean & Split (80/20, chronological) → Feature Engineering
   → Train RF vs XGBoost (per station) → Calibrate Threshold → Export
   → Streamlit Dashboard + Gemini Agent
```

---

## ✨ Features

- 🏭 8-station synthetic factory line, minute-level data, reproducible (fixed seed)
- 🎯 Per-station model, calibrated to a station-specific Recall floor (80%–90%)
- 🔍 Explainable alerts — names the real cause *and* the sensor that triggered it
- ⏱️ Time-travel dashboard — scrub through the test period, jump to any date/time
- 💬 Gemini tool-calling agent — 8 tools, zero hallucinated numbers (every answer comes from a real function call)
- 🛡️ Leak-free by design — sensor selection never touches the test set

---

## 📊 Key Results

Best model: **XGBoost**, selected per station by Precision at a fixed Recall floor (Random Forest trained and compared, but XGBoost wins Precision in all 8 stations).

| Metric | Value |
|---|---|
| Average Accuracy | ≈ 94.1% |
| Average Precision | ≈ 74.1% (range 70.7%–78.0%) |
| Recall | 80%–90% per station (station-specific floor) |

**Why Mechanical/Electrical only?** Validated empirically, not assumed: combining all 6 downtime causes into one target collapsed average Recall from ~90% to ~49%, because "positive" minutes jumped from ~16% to ~72% of the dataset.

**Data leakage caught and fixed:** sensor thresholds were originally computed on each station's full history, including the future test window — fixed by using the training split only.

**Why Recall floors differ by station:** stations with fewer or less informative sensors can't sustain a high Recall without Precision collapsing. ST7 has only one sensor (`energy_consumption`); forcing the same 90% Recall floor used elsewhere crashes its Precision to **12.8%** — versus **72.5%** at its actual, station-specific 80% floor. This was tested directly, not assumed: it's the project's key "biggest challenge" finding.

---

## 🗂️ Project Structure

```
project/
│
├── app.py                       # Streamlit dashboard + Gemini tool-calling agent
├── functions_api.py             # Reusable prediction/explanation logic
├── FinalProject_PM_opt.ipynb    # Full pipeline: simulate → clean → engineer → train → evaluate
├── factory_simulator.ipynb      # Synthetic 8-station factory data generator
├── trained_models/
│   └── trained_models.pkl       # Slim per-station artifact (winning model only)
├── source/output/               # Simulator CSV outputs
├── images_hmi/                  # Dashboard station icons
├── FPM.pptx                     # Project presentation
├── requirements.txt
└── README.md
```

---

## 🧩 Components

### 🏭 Factory Simulator
Generates 30 days × 8 stations of minute-level sensor + downtime data, with a physical failure model per station (`factory_simulator.ipynb`).

### 📈 Feature Engineering
Causal, rolling-window features per sensor — `mean15`, `std15`, `slope15`, `curr` — with a 30-minute-ahead failure target, computed with no test-set leakage.

### 🤖 Per-Station Models
Random Forest vs. XGBoost, trained and evaluated independently for each of the 8 stations, with a decision threshold calibrated via the Precision-Recall curve at a station-specific Recall floor.

### 💬 Gemini Tool-Calling Agent
8 Python tools — the LLM only decides *what* to call and how to phrase the answer, never computes anything itself:
- `predict_station_risk(station)` — current failure probability
- `get_prob_fail_sensor(station)` — sensor closest to its failure threshold right now
- `get_top_failing_sensor()` — across all 8 stations, which sensor has historically triggered the most failures
- `get_feature_importance(station)` — top features the station's model relies on globally
- `get_station_sensors(station)`, `get_sensor_threshold(station, sensor)` — model transparency
- `get_downtime_history(station, cause)` — aggregate downtime stats
- `get_model_performance(station)` — Recall/Precision/F1 for the deployed model

### 🖥️ Streamlit App
Time-travel slider over the held-out test period, per-station risk cards with explainable failure messages, and a chat panel wired to the agent above.


## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python, pandas | Core language & data handling |
| scikit-learn, XGBoost | Model training & evaluation |
| Google Gemini (Interactions API) | Tool-calling agent |
| Streamlit | Interactive dashboard |
| Matplotlib/Seaborn | EDA & results charts |

---

## ⚠️ Known Limitations

- Trained entirely on **synthetic** simulator data — real-machine sensor drift/wear is not represented
- Only Mechanical/Electrical failures are predicted; other stoppage causes are shown as ground truth but out of scope (validated empirically — see *Key Results*)
- Stations with very few sensors (e.g., ST7, 1 sensor) cap how high a Recall floor can go before Precision becomes unusable — this is a real sensor-coverage limit, not a modeling choice
- `get_prob_fail_sensor` is a similarity heuristic (distance to a historical failure threshold), not a trained classifier — too few labeled events per sensor to train one reliably
- A 5-configuration hyperparameter search gave only marginal, inconsistent gains over the hand-chosen defaults
- Gemini free-tier rate limits (5 req/min) can still be hit under heavy chat use; the app retries once, then fails gracefully

---

## 🔮 Next Steps

- Stream real sensor data from the PLC/SCADA historian (periodic CSV export) instead of the offline simulator; retrain on a schedule using the same pipeline, promoting a new model only if it beats the current one (champion/challenger) on the same test set
- Add an unsupervised anomaly-detection model (e.g., IsolationForest) alongside the supervised model, to catch concept drift the supervised model alone would miss
- True future holdout: simulate additional unseen days beyond the training window
- Model Blocked/Starved stoppages with line-balancing (neighboring-station) features
- Optional: voice-enabled assistant for hands-free shop-floor use

---

# ⚡ JIGNASA 2026 — Campus Electricity Load Planning Assistant

### JIG26_04 | Track A — ANN & Predictive Analytics

An AI-powered **electricity load forecasting and budget planning system** that predicts the next 24 hours of campus electricity demand, estimates energy costs, tracks budget utilization, and enables What-If scenario analysis — all powered by a verified 579-feature deep learning pipeline.

---

## 🎯 Problem Statement

Campus electricity demand fluctuates significantly due to:

- **Temporal patterns** — time of day, day of week, month
- **Academic activity** — lectures, exams (mid-sem / end-sem), library usage
- **Occupancy** — academic buildings, hostels, mess halls
- **Weather** — temperature, humidity, wind speed
- **Historical load** — recent consumption trends, weekly seasonality

Without accurate forecasting, campus administrators cannot proactively plan for peak demand, budget electricity costs, or identify energy-saving opportunities.

**JIGNASA solves this** by providing a live 24-hour load forecast with automated budget tracking, cost estimation, peak-demand alerts, and actionable planning insights.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🔮 **24-Hour Forecast** | Predicts hourly electricity load for the next 24 hours |
| 💰 **Budget & Cost Planning** | Configurable energy budget (kWh) and cost budget (₹) with real-time utilization tracking |
| 🚦 **3-Tier Status System** | 🟢 Comfortable (<90%) · 🟡 Near Limit (90–100%) · 🔴 Exceeded (>100%) |
| 🔬 **What-If Scenarios** | Modify occupancy/exam variables and rerun the full ANN pipeline to compare energy & cost impact |
| 📊 **Hourly Breakdown Table** | Detailed per-hour kW predictions with timestamps and scenario comparison |
| 🌦️ **Live Weather Integration** | Pulls real-time weather from Open-Meteo API for accurate context |
| 📅 **Auto Calendar Detection** | Automatically derives Indian holidays, weekends, and working days |
| ⚡ **Peak Demand Alerts** | Flags hours exceeding the 994.4 kW planning threshold |
| 🧠 **Planning Insights** | Context-aware recommendations based on utilization, peak timing, and contributing factors |
| 🕐 **Live Mode** | Accepts any date (including future/today) by intelligently mapping to historical proxy weeks |

---

## 🧠 Model Architecture

The forecasting engine is a **Multi-Layer Perceptron (MLP)** built with TensorFlow/Keras.

```
579 Input Features
      ↓
Dense(128) → ReLU → Dropout
      ↓
Dense(64)  → ReLU → Dropout
      ↓
Dense(32)  → ReLU
      ↓
24 Output Neurons (hourly kW predictions)
```

### 579-Feature Pipeline Breakdown

| Category | Count | Examples |
|---|---|---|
| Historical load features | 7 | `load_now`, `load_lag_1`, `load_lag_24`, `load_lag_168`, `rolling_mean_24`, `rolling_mean_168`, `rolling_std_24` |
| Current temporal/context | 20 | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `is_weekend`, `is_holiday`, occupancy, weather, exam flags |
| Future features (24h × 23) | 552 | Forecast weather, calendar, occupancy, temporal encodings per future hour |
| **Total** | **579** | |

### Verified Performance (Chronological Test Set: May–Dec 2017)

| Metric | Value |
|---|---|
| **Overall MAE** | 35.30 kW |
| **Overall RMSE** | 44.51 kW |
| Peak MAE | 44.47 kW |
| Peak RMSE | 54.50 kW |
| Off-Peak MAE | 33.81 kW |
| Off-Peak RMSE | 42.66 kW |

---

## 🖥️ Dashboard Overview

The web UI provides a unified planning dashboard with:

1. **Top Metric Cards** — Predicted Energy (kWh), Estimated Cost (₹), Peak Load & Time, Budget Status
2. **Planning Insights Banner** — Context-aware alerts with actionable recommendations
3. **24-Hour Forecast Chart** — Interactive SVG with baseline (blue) and scenario (amber) overlays + peak threshold line
4. **Hourly Breakdown Table** — Scrollable table with per-hour timestamps, baseline load, and scenario comparison deltas
5. **Sidebar Controls** — Date picker, budget configuration (tariff, energy budget, cost budget, warning threshold), and What-If scenario inputs

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| ML Framework | TensorFlow / Keras |
| Feature Scaling | scikit-learn (StandardScaler) |
| Frontend | HTML5, Tailwind CSS, Vanilla JS |
| Weather API | Open-Meteo (free, no API key) |
| Calendar | `holidays` Python library (India) |
| Data | IIIT-Delhi campus electricity dataset (2016–2017) |

---

## 📦 Quick Start

### Prerequisites

```bash
pip install flask tensorflow pandas numpy scikit-learn joblib requests holidays openpyxl
```

### Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

### What You'll See

1. The dashboard auto-loads with **today's date** and runs a live inference
2. Adjust the **Budget Configuration** (tariff, daily energy budget, cost budget) in the sidebar
3. Click **Run Baseline Inference** to forecast any date
4. Modify occupancy/exam settings and click **Simulate What-If** to compare scenarios

---

## 📁 Project Structure

```
├── app.py                          # Flask backend with ANN inference + What-If pipeline
├── templates/
│   └── v5.html                     # Full dashboard UI (Tailwind + vanilla JS)
├── op/
│   ├── ann_24h.keras               # Trained 579→24 ANN model
│   ├── scalers.joblib              # Feature scaler + target mean/std
│   └── config.json                 # Feature pipeline configuration
├── ultimate_campus_dataset.csv     # Master dataset for historical context
├── verify_ann.py                   # Model verification & evaluation script
├── dataset_builder.py              # Dataset construction pipeline
├── build_final_features.py         # 579-feature engineering pipeline
└── README.md
```

---

## 👥 Team

**JIGNASA 2026 — JIG26_04**

---

## 📜 License

This project was built for the JIGNASA 2026 Hackathon. All rights reserved.

# ⚡ Electricity Load Planning Assistant

### JIGNASA 2026 — JIG26_04 | Track A — ANN & Predictive

An AI-based electricity load forecasting system that predicts the **next 24 hours of campus electricity demand** using historical load, temporal, academic, occupancy, and weather-related features.

---

## 🎯 Problem

Electricity demand on a campus changes throughout the day and week due to factors such as:

- Time of day
- Day of the week
- Academic activity
- Examinations
- Occupancy
- Weather conditions
- Previous electricity consumption

The goal is to provide a **24-hour load forecast** that can help with electricity planning and peak-demand awareness.

---

## 🚀 Key Features

- 🔮 24-hour electricity load forecasting
- 🧠 Neural-network-based forecasting using an MLP/ANN
- ⏱️ Lag features for recent and weekly load patterns
- 📊 Rolling statistics for recent demand trends
- 📅 Calendar and academic features
- 🌦️ Weather-related features
- 👥 Occupancy-related features
- 📈 Peak-demand detection
- 📉 MAE and RMSE evaluation
- 🔬 Comparison with simple forecasting baselines
- 🛠️ Outage/missing-data handling
- 📊 Forecast visualization and planning dashboard

---

## 🧠 Model

The forecasting model is a **Multi-Layer Perceptron (MLP)** implemented using PyTorch.

### Architecture

```text
Input Features
      ↓
Dense Layer (128)
      ↓
ReLU + Dropout
      ↓
Dense Layer (64)
      ↓
ReLU + Dropout
      ↓
Dense Layer (32)
      ↓
ReLU
      ↓
24 Output Values

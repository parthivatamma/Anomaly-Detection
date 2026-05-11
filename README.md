# AI-Orchestrated Anomaly Detection Engine

A modernization of a legacy anomaly detection project (circa 2022) into a production-ready diagnostic dashboard. This application uses the **Isolation Forest** method to identify outliers in telemetry data and leverages a **local LLM (Llama 3 via Ollama)** to generate human-readable technical diagnostics for the identified anomalies.

## 🚀 The Vision: "Legacy to AI"

This project demonstrates the evolution of a data science pipeline—taking raw numerical analysis and wrapping it in a modern, interactive UI with an AI "Analyst" layer. It highlights the transition from static scripts to dynamic, orchestrated systems.

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Modeling:** Scikit-Learn (Isolation Forest)
- **Orchestration:** Ollama (Llama 3)
- **Visualization:** Plotly, Streamlit
- **Data Handling:** Pandas, NumPy

## ✨ Key Features

- **Dynamic Parameter Tuning:** Real-time adjustment of 'Contamination' levels and model estimators via the Streamlit sidebar.
- **Automated Incident Analysis:** When an anomaly is detected, the system extracts the data and queries a local LLM to hypothesize mechanical failure points (e.g., automotive sensor failure or hardware degradation).
- **Interactive Telemetry:** Full visibility into data distribution and outliers through interactive scatter plots and histograms.

## 🚦 Getting Started

### Prerequisites

1. Install [Ollama](https://ollama.com/) and download the model:
   ```bash
   ollama pull llama3
   ```

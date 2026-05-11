import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
import ollama

st.set_page_config(page_title="Anomaly Detection Engine", layout="wide")
st.title("Interactive Anomaly Detection (Isolation Forest)")

# ==========================================
# 1. Data Generation (Replacing fileFiller.py)
# ==========================================
# We use a session state so it doesn't regenerate on every slider click
if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame()


def generate_data():
    """Generates synthetic scores for 100 overs."""
    # Simulate mostly normal scores, with occasional high outliers
    normal_scores = np.random.randint(0, 15, size=90)
    outlier_scores = np.random.randint(25, 37, size=10)  # 36 is max in an over

    all_scores = np.concatenate([normal_scores, outlier_scores])
    np.random.shuffle(all_scores)  # Mix them up

    df = pd.DataFrame({"Overs": range(1, 101), "Scores": all_scores})
    st.session_state.dataset = df


if st.button("Generate New Synthetic Data"):
    generate_data()

# Ensure we have data to work with
if st.session_state.dataset.empty:
    generate_data()

df = st.session_state.dataset.copy()

# ==========================================
# 2. Model Parameters (Replacing Hardcoded Variables)
# ==========================================
st.sidebar.header("Model Tuning")
# Here we fix the hardcoded 0.2 contamination
contamination = st.sidebar.slider(
    "Contamination Level", min_value=0.01, max_value=0.30, value=0.20, step=0.01
)
n_estimators = st.sidebar.slider(
    "Number of Estimators", min_value=10, max_value=100, value=50, step=10
)

# ==========================================
# 3. Model Execution (Replacing AnomalyDetection.py)
# ==========================================
model = IsolationForest(
    n_estimators=n_estimators, contamination=contamination, random_state=42
)

# Fit and predict
model.fit(df[["Scores"]])
df["Anomaly_Score"] = model.decision_function(df[["Scores"]])
df["Prediction"] = model.predict(df[["Scores"]])

# Map the -1/1 to readable labels
df["Status"] = df["Prediction"].map({1: "Normal", -1: "Anomaly"})

# ==========================================
# 4. Interactive Visualization
# ==========================================
st.subheader("Data Visualization")

# Interactive Scatter Plot replacing plt.plot()
fig = px.scatter(
    df,
    x="Overs",
    y="Scores",
    color="Status",
    color_discrete_map={"Normal": "#1f77b4", "Anomaly": "#d62728"},
    title=f"Anomaly Detection (Total Outliers: {len(df[df['Status'] == 'Anomaly'])})",
)
st.plotly_chart(fig, width="stretch")

# Interactive Histogram replacing plt.hist()
fig_hist = px.histogram(
    df, x="Scores", color="Status", nbins=20, title="Distribution of Scores"
)
st.plotly_chart(fig_hist, width="stretch")

# ==========================================
# 5. The LLM Diagnostics Placeholder
# ==========================================
st.divider()
st.subheader("🤖 AI Technical Diagnostics")

# Filter for anomalies
anomalies = df[df["Status"] == "Anomaly"]

if not anomalies.empty:
    # Let the user pick which anomaly to analyze from a dropdown
    anomaly_list = anomalies["Overs"].tolist()
    selected_over = st.selectbox(
        "Select an Anomaly to Analyze (by Over #):", anomaly_list
    )

    # Get the specific data for that over
    target_row = anomalies[anomalies["Overs"] == selected_over].iloc[0]

    if st.button("Generate Diagnostic Report"):
        with st.spinner("Analyzing telemetry with local LLM..."):
            try:
                # This prompt is tailored to your interests (ASU engineering/Automotive)
                # It asks the LLM to act as a Senior Diagnostic Engineer
                prompt = f"""
                You are a Senior Diagnostic Engineer. You are reviewing a 'Scores per Over' dataset 
                where most values represent standard operating metrics, but a significant 
                outlier has been detected by an Isolation Forest model.

                ANOMALY DATA:
                - Over: {target_row['Overs']}
                - Recorded Score: {target_row['Scores']}
                - Model Anomaly Score: {target_row['Anomaly_Score']:.4f}

                TASK:
                Provide a 3-sentence technical analysis. 
                1. Identify the statistical significance of this deviation.
                2. Hypothesize a potential 'real-world' cause (e.g., if this were an 
                   M276 engine sensor or a mechanical keyboard switch failure).
                3. Suggest a corrective action.
                
                Keep the tone professional and concise.
                """

                response = ollama.chat(
                    model="llama3", messages=[{"role": "user", "content": prompt}]
                )

                # Display the result in a nice box
                st.markdown("### Diagnostic Report")
                st.success(response["message"]["content"])

            except Exception as e:
                st.error(
                    f"Could not connect to Ollama. Make sure the Ollama app is running. Error: {e}"
                )
else:
    st.info(
        "No anomalies detected to analyze. Try increasing the 'Contamination' slider."
    )

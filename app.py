#!/usr/bin/env python3
"""
app.py  –  Streamlit demo for Disease-Symptom Risk Prediction
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib

# ── Load model & encoder ───────────────────────────────────────────
MODEL_PATH = "symptom_model.joblib"
ENC_PATH   = "label_encoder.joblib"
DB_FILE    = "disease_symptoms.db"
TABLE      = "symptoms"

model = joblib.load(MODEL_PATH)
le    = joblib.load(ENC_PATH)
engine = create_engine(f"sqlite:///{DB_FILE}")

# ── Get symptom column list (dynamic, so it always matches training) ─
symptom_cols = pd.read_sql("PRAGMA table_info(symptoms);",
                           engine)["name"].tolist()
symptom_cols.remove("prognosis")  # target column

# ── UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disease-Risk Predictor", layout="centered")
st.title("🩺 Disease-Symptom Risk Predictor")

st.write("Tick the symptoms you observe, then click **Predict**.")

# Show check-boxes in 3 columns for compactness
cols = st.columns(3)
user_flags = {}
for i, col in enumerate(symptom_cols):
    with cols[i % 3]:
        user_flags[col] = st.checkbox(col.replace('_', ' ').capitalize())

if st.button("Predict", type="primary"):
    # Build a single-row DataFrame in exact column order
    df_input = pd.DataFrame([[user_flags[c] for c in symptom_cols]],
                            columns=symptom_cols)

    # Predict
    probs = model.predict_proba(df_input)[0]
    pred_class = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_class])[0]
    st.success(f"**Most likely diagnosis: {pred_label}**")

    # Probability table (top-10)
    top_idx = np.argsort(probs)[::-1][:10]
    top_df  = pd.DataFrame({
        "Disease":     le.inverse_transform(top_idx),
        "Probability": probs[top_idx]
    })
    st.write("### Full probability breakdown (top-10)")
    st.dataframe(top_df.style.format({"Probability": "{:.2%}"}), hide_index=True)

    st.caption("Powered by scikit-learn · Data courtesy of kaushil268 / Kaggle")



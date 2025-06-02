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
# This 'symptom_cols_original_order' now holds the exact order from the DB/training
symptom_cols_original_order = pd.read_sql("PRAGMA table_info(symptoms);",
                                          engine)["name"].tolist()
symptom_cols_original_order.remove("prognosis")  # target column

# Create a separate list for display, sorted alphabetically
display_symptom_cols = sorted(symptom_cols_original_order) # Use sorted() to create a new list

# ── UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disease-Risk Predictor", layout="wide")
st.title("🩺 Disease-Symptom Risk Predictor")

# --- UPDATED DISCLAIMER CODE HERE ---
st.warning("""
    **Disclaimer:** This tool is for **informational purposes only** and should **NOT** be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Do not disregard professional medical advice or delay in seeking it because of something you have read on this tool.
    
    **This application was developed as a project for a UofM Data Analytics Bootcamp and is intended for demonstration purposes only.**
""")

st.write("Tick the symptoms you observe, then click **Predict**.")

# Show check-boxes in 4 columns for compactness
cols = st.columns(4)
user_flags = {}
for i, col in enumerate(display_symptom_cols):
    with cols[i % 4]:
        user_flags[col] = st.checkbox(col.replace('_', ' ').capitalize())

if st.button("Predict", type="primary"):
    # Build a single-row DataFrame in exact column order
    df_input = pd.DataFrame([[user_flags[c] for c in symptom_cols_original_order]],
                            columns=symptom_cols_original_order)

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



import streamlit as st
import pandas as pd
import joblib
import json
import os

# --- Paths ---
DATA_DIR = "Data"
LOGREG_PATH = os.path.join(DATA_DIR, "log_reg_model.pkl")
RF_PATH = os.path.join(DATA_DIR, "rf_model.pkl")
RESULTS_PATH = os.path.join(DATA_DIR, "training_results.json")
SCHEMA_PATH = os.path.join(DATA_DIR, "input_schema.json")

# --- Load Models and Schema ---
log_reg = joblib.load(LOGREG_PATH)
rf = joblib.load(RF_PATH)

with open(RESULTS_PATH, "r") as f:
    training_results = json.load(f)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="wide")
st.title("🫀 Heart Disease Prediction")
st.write("Provide health details to check the risk of heart disease using Logistic Regression and Random Forest models.")

# --- Build Input Form ---
input_dict = {}

# Numeric inputs (side by side)
st.subheader("Numeric Information")
cols = st.columns(len(schema["numeric"]))
for i, col in enumerate(schema["numeric"]):
    with cols[i]:
        input_dict[col] = st.number_input(col, step=1.0)

# Categorical inputs (grouped)
st.subheader("Categorical Information")
cat_cols = list(schema["categorical"].keys())
cols = st.columns(3)
for i, col in enumerate(cat_cols):
    with cols[i % 3]:
        options = [""] + schema["categorical"][col]
        input_dict[col] = st.selectbox(col, options)

# --- Prediction Button ---
if st.button("Predict"):
    # Check for missing fields
    if "" in input_dict.values():
        st.warning("⚠️ Please fill in all fields before prediction.")
    else:
        input_df = pd.DataFrame([input_dict])

        # Logistic Regression
        log_pred = log_reg.predict(input_df)[0]
        log_prob = log_reg.predict_proba(input_df)[0][1]

        # Random Forest
        rf_pred = rf.predict(input_df)[0]
        rf_prob = rf.predict_proba(input_df)[0][1]

        st.subheader("📈 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Logistic Regression")
            if log_pred == "Yes":
                st.error(f"⚠️ Predicted: Heart Disease (Probability: {log_prob:.2f})")
            else:
                st.success(f"✅ Predicted: No Heart Disease (Probability: {log_prob:.2f})")

        with col2:
            st.markdown("### Random Forest")
            if rf_pred == "Yes":
                st.error(f"⚠️ Predicted: Heart Disease (Probability: {rf_prob:.2f})")
            else:
                st.success(f"✅ Predicted: No Heart Disease (Probability: {rf_prob:.2f})")

# --- Training Metrics Section ---
# st.subheader("Model Training Performance")
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("#### Logistic Regression")
#     st.json(training_results["logistic_regression"])

# with col2:
#     st.markdown("#### Random Forest")
#     st.json(training_results["random_forest"])

# --- Explanatory Note ---
st.markdown("---")
st.info(
    "💡 **Note:** This is a clear practical case of **class imbalance**. "
    "The dataset contains many more 'No Heart Disease' cases than 'Yes'. "
    "As a result, the models may lean towards predicting 'No'. "
    "This explains why some patients labeled as 'Yes' in the dataset "
    "are predicted as 'No' by both models. "
    "Class imbalance reduces recall for the positive class, but probabilities "
    "still provide useful insights. Adjusting the decision threshold or "
    "using balanced training techniques can help improve detection."
)
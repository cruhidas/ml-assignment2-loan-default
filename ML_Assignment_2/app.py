import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Predictor", layout="centered")

st.title("Loan Default (LendingClub) - Model Demo")
st.write("Student demo app for ML Assignment 2. Upload a CSV (test data) and pick a model.")

MODEL_DIR = "saved_models"

@st.cache_resource
def load_assets():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))

    models = {}
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".joblib") and fname not in ["scaler.joblib", "feature_columns.joblib"]:
            key = fname.replace(".joblib", "").replace("_", " ").title()
            models[key] = joblib.load(os.path.join(MODEL_DIR, fname))
    return scaler, feat_cols, models

def preprocess_for_app(df_raw: pd.DataFrame, feature_cols):
    df = df_raw.copy()

    # If user uploaded loan_status, convert; else if 'target' exists, use that.
    if "loan_status" in df.columns and "target" not in df.columns:
        df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
        df["target"] = df["loan_status"].map({"Fully Paid": 0, "Charged Off": 1}).astype(int)
        df.drop(columns=["loan_status"], inplace=True)

    # Convert common percent columns if exist
    for c in ["int_rate", "revol_util"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace("%","")
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic term/emp_length parsing
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    if "emp_length" in df.columns:
        def parse_emp(x):
            x = str(x)
            if x.strip().startswith("<"):
                return 0
            import re
            m = re.search(r"(\d+)", x)
            return int(m.group(1)) if m else 0
        df["emp_length"] = df["emp_length"].apply(parse_emp)

    # Fill missing quickly (simple)
    for col in df.columns:
        if col == "target":
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else "NA")
        else:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["target"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    # Align columns to training feature set
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    y = df["target"] if "target" in df.columns else None
    return X, y

def show_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_title("Confusion Matrix")
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center')
    st.pyplot(fig)

# ---- UI ----
uploaded = st.file_uploader("Upload test CSV (preferably contains target column as 'target' or 'loan_status')", type=["csv"])

if not os.path.exists(MODEL_DIR):
    st.error("saved_models folder not found. Run train_models.py first.")
    st.stop()

scaler, feat_cols, models = load_assets()

if not models:
    st.error("No models found in saved_models. Run train_models.py first.")
    st.stop()

model_name = st.selectbox("Pick a model", sorted(models.keys()))
model = models[model_name]

if uploaded is not None:
    df_in = pd.read_csv(uploaded, low_memory=False)
    st.write("Uploaded shape:", df_in.shape)
    st.dataframe(df_in.head(5))

    X, y = preprocess_for_app(df_in, feat_cols)

    # Choose scaled vs not scaled based on model name (simple rule)
    need_scaled = any(k in model_name.lower() for k in ["logistic", "knn", "naive bayes"])
    if need_scaled:
        X_use = scaler.transform(X)
        if "naive bayes" in model_name.lower():
            X_use_for_pred = X_use.toarray()
        else:
            X_use_for_pred = X_use
    else:
        X_use_for_pred = X

    # Predict
    y_pred = model.predict(X_use_for_pred)

    # Probabilities for AUC
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_use_for_pred)[:,1]

    st.subheader("Predictions")
    st.write(pd.Series(y_pred).value_counts().rename("count"))

    if y is not None:
        st.subheader("Evaluation (only if uploaded CSV contains target)")
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y, y_pred)
        auc = roc_auc_score(y, y_prob) if y_prob is not None else roc_auc_score(y, y_pred)

        st.write({
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

        show_conf_matrix(y, y_pred)

        st.text("Classification Report:")
        st.text(classification_report(y, y_pred, zero_division=0))
    else:
        st.info("No target found in uploaded CSV. Add a 'target' column (0/1) to see metrics.")
else:
    st.caption("Tip: After training, you can upload the same dataset but keep only a small sample for the Streamlit free tier.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# Try LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

st.set_page_config("ML Ops Tool", layout="wide")
st.title("ML Ops Tool")

# ------------------ Upload ------------------

file = st.file_uploader(
    "📂 Upload your CSV file",
    type=["csv"],
    help="Drag and drop a CSV file to begin"
)

if file is None:
    st.markdown(
        """
        <div style='text-align:center; padding: 60px; border: 2px dashed #888; border-radius: 12px;'>
            <h2>📊 ML Ops Tool</h2>
            <p style='font-size:18px'>
                Upload a CSV file to start building models
            </p>
            <p>
                Supported models:
                <br>• Linear Regression
                <br>• Logistic Regression
                <br>• Gradient Boosting
                <br>• Neural Networks
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

df = pd.read_csv(file)
predict_only = st.sidebar.toggle("Predict missing targets only", value=False)
st.dataframe(df.head())

model_container = st.sidebar.container()

# ------------------ Column Selection ------------------
target = st.sidebar.selectbox("Target column", df.columns)
features = st.sidebar.multiselect(
    "Feature columns",
    [c for c in df.columns if c != target],
    default=[c for c in df.columns if c != target][:5]
)

if len(features) == 0:
    st.stop()

X = df[features]
y = df[target]

# Drop rows where target is NaN (REQUIRED for sklearn)
# split rows based on missing target
train_mask = y.notna()

X_train_all = X.loc[train_mask]
y_train_all = y.loc[train_mask]

X_to_predict = X.loc[~train_mask]

# Detect type only from training rows
is_binary = y_train_all.nunique() == 2

# ------------------ Model Selection ------------------
with model_container:
    if is_binary:
        model_choice = st.selectbox(
            "Model",
            ["Logistic Regression", "GBM", "Neural Network"]
        )
    else:
        model_choice = st.selectbox(
            "Model",
            ["Linear Regression", "GBM", "Neural Network"]
        )

# ------------------ Preprocessing ------------------
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Version-safe OneHotEncoder
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ]), cat_cols)
])

# ------------------ Train/Test ------------------
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)
handle_imbalance = st.sidebar.toggle(
    "Handle class imbalance (recommended for rare classes)",
    value=True
)
X_train, X_test, y_train, y_test = train_test_split(
    X_train_all, y_train_all, test_size=test_size/100, random_state=42
)

# ------------------ Model Init ------------------
if model_choice == "Linear Regression":
    model = Ridge()

elif model_choice == "Logistic Regression":
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced" if handle_imbalance else None
    )


elif model_choice == "GBM":
    if HAS_LGB:
        model = lgb.LGBMClassifier() if is_binary else lgb.LGBMRegressor()
    else:
        if is_binary:
            try:
                model = HistGradientBoostingClassifier(
                    class_weight="balanced" if handle_imbalance else None
                )
            except:
                model = HistGradientBoostingClassifier()
        else:
            model = HistGradientBoostingRegressor()



else:  # Neural Network
    model = (
        MLPClassifier(max_iter=500)
        if is_binary else
        MLPRegressor(max_iter=500)
    )

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# ------------------ Train ------------------
if st.button("Train Model"):
    pipeline.fit(X_train, y_train)

    # evaluation predictions
    preds = pipeline.predict(X_test)

    # predict missing targets
    future_preds = None
    if len(X_to_predict) > 0:
        future_preds = pipeline.predict(X_to_predict)

    col1, col2 = st.columns([1, 2])

    if is_binary:

        if not hasattr(pipeline, "predict_proba"):
            st.warning("This model does not support probability scores.")
            proba = None
        else:
            proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, proba)
        }
        with col1:
            st.subheader("Metrics")
            st.json(metrics)

            st.subheader("Model Summary")

            if is_binary:
                if metrics["Recall"] < 0.6:
                    st.warning("Low recall — model is missing many positives. Be careful.")
                elif metrics["ROC AUC"] < 0.7:
                    st.info("Model is weak. Consider adding more features or data.")
                else:
                    st.success("Model looks solid and usable.")
            else:
                pass
        cm = confusion_matrix(y_test, preds)
        with col2:
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(cm))

    else:
        mse = mean_squared_error(y_test, preds)
        metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, preds)
        }
        with col1:
            st.subheader("Metrics")
            st.json(metrics)

            st.subheader("Model Summary")

            if metrics["R2"] < 0.4:
                st.warning("Very weak model — predictions are unreliable.")
            elif metrics["R2"] < 0.7:
                st.info("Okay model — usable but improve if possible.")
            else:
                st.success("Strong model — predictions are quite reliable.")

            st.subheader("🔎 Model Health Check")

            r2 = metrics["R2"]

            if r2 < 0.4:
                st.error("❗Low explanatory power — model is not explaining the data well.")
            elif r2 < 0.7:
                st.warning("⚠️ Medium strength — good for trends, not precise forecasts.")
            else:
                st.success("💡 Strong model — explains most of the variation.")

        st.markdown("---")

        left, center, right = st.columns([1, 3, 1])

        with center:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(preds, y_test - preds, alpha=0.6)
            ax.axhline(0, color="red", linewidth=1)
            ax.set_title("Residuals")
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals (Actual - Predicted)")
            st.pyplot(fig, use_container_width=True)

            # ======================================
            # FEATURE IMPORTANCE (GBM ONLY)
            # ======================================
            if model_choice == "GBM":
                try:
                    model_step = pipeline.named_steps["model"]

                    if hasattr(model_step, "feature_importances_"):
                        st.subheader("Feature Importance (GBM)")

                        importances = model_step.feature_importances_

                        fig_imp, ax_imp = plt.subplots()
                        ax_imp.bar(range(len(importances)), importances)
                        ax_imp.set_title("GBM Feature Importance")
                        st.pyplot(fig_imp)

                    else:
                        st.info("This GBM version does not expose feature importances.")

                except Exception as e:
                    st.info(f"Feature importance unavailable: {e}")
    # ------------------ Save Outputs ------------------
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_choice.replace(' ','_')}.joblib"
    joblib.dump(pipeline, model_path)

    st.success(f"Model saved: {model_path}")

    # test predictions
    # test part (trained rows)
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["y_pred"] = preds
    out["Row_Type"] = "train"

    # prediction-only rows
    if future_preds is not None:
        future = X_to_predict.copy()
        future["y_true"] = None
        future["y_pred"] = future_preds
        future["Row_Type"] = "predict"

        out = pd.concat([out, future], axis=0)

    if predict_only:
        out = out[out["Row_Type"] == "predict"]
    st.download_button(
        "Download Predictions CSV",
        out.to_csv(index=False),
        "predictions.csv"
    )

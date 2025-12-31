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
from sklearn.inspection import permutation_importance

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
        threshold_mode = st.sidebar.selectbox(
            "Threshold Mode",
            ["Manual", "Optimize for Recall", "Optimize for Precision", "Optimize F1"]
        )

        threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01)

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
suggest_drops = st.sidebar.toggle("Suggest features to drop", value=False)

# ------------------ Automated Imbalance Handling (Visualized) ------------------
if is_binary:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Imbalance Handling")

    # --- 1. SETTINGS & AUTOMATION ---

    # User decides threshold
    user_threshold = st.sidebar.slider(
        "Auto-balance Threshold (%)",
        min_value=5,
        max_value=50,
        value=15,
        help="If the minority class is smaller than this %, balancing will happen automatically."
    ) / 100.0

    # Calculate actual ratio
    counts = y_train_all.value_counts()
    actual_ratio = y_train_all.value_counts(normalize=True).min()

    # Logic to decide if we force it ON or let user choose
    is_auto_enabled = False

    if actual_ratio < user_threshold:
        st.sidebar.warning(f"⚠️ Imbalance Detected! (Minority: {round(actual_ratio * 100, 1)}%)")
        handle_imbalance = True
        is_auto_enabled = True
    else:
        st.sidebar.success(f"Data is balanced. (Minority: {round(actual_ratio * 100, 1)}%)")
        handle_imbalance = st.sidebar.toggle(
            "Handle class imbalance",
            value=False
        )

    # --- 2. VISUAL CONFIRMATION (The new part) ---

    st.sidebar.markdown("#### 📊 Current Class Distribution")

    # Create a clean DataFrame for display
    balance_df = pd.DataFrame({
        "Count": counts,
        "Percentage": (y_train_all.value_counts(normalize=True) * 100).round(1).astype(str) + "%"
    })

    # Show Table
    st.sidebar.dataframe(balance_df, use_container_width=True)

    # Show Chart (Simple Bar)
    st.sidebar.bar_chart(counts, color="#4b72af", height=150)

    # --- 3. FINAL STATUS INDICATOR ---
    st.sidebar.markdown("#### 🛠️ Status")

    if handle_imbalance:
        if is_auto_enabled:
            st.sidebar.success("✅ Balancing: **ACTIVE (Auto-Forced)**")
        else:
            st.sidebar.success("✅ Balancing: **ACTIVE (Manual)**")
    else:
        st.sidebar.info("ℹ️ Balancing: **OFF**")

else:
    # Not binary? Turn it off implicitly.
    handle_imbalance = False

X_train, X_test, y_train, y_test = train_test_split(
    X_train_all, y_train_all, test_size=test_size / 100, random_state=42
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
        # Use LightGBM if installed
        if is_binary:
            model = lgb.LGBMClassifier(
                random_state=42,
                class_weight="balanced" if handle_imbalance else None
            )
        else:
            model = lgb.LGBMRegressor(random_state=42)
    else:
        # Fallback to Scikit-Learn if LightGBM is missing
        if is_binary:
            try:
                model = HistGradientBoostingClassifier(
                    class_weight="balanced" if handle_imbalance else None,
                    random_state=42
                )
            except TypeError:
                # Handle older sklearn versions without class_weight support
                model = HistGradientBoostingClassifier(random_state=42)
        else:
            model = HistGradientBoostingRegressor(random_state=42)

else:  # Neural Network
    model = (
        MLPClassifier(max_iter=500, random_state=42)
        if is_binary else
        MLPRegressor(max_iter=500, random_state=42)
    )

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# ------------------ Train ------------------
if st.button("Train Model"):
    pipeline.fit(X_train, y_train)
    # default predictions (used for regression)
    preds = pipeline.predict(X_test)

    # override class decisions based on threshold
    if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
        # Sort unique values to ensure 0 is first, 1 is second consistently
        classes = sorted(y_test.unique())
        y_bin = (y_test == classes[1]).astype(int)

        proba = pipeline.predict_proba(X_test)[:, 1]
        from sklearn.metrics import precision_recall_curve

        if threshold_mode == "Manual":
            best_threshold = threshold

        else:
            precisions, recalls, ths = precision_recall_curve(y_bin, proba)

            if threshold_mode == "Optimize for Recall":
                idx = recalls.argmax()
                best_threshold = ths[idx]

            elif threshold_mode == "Optimize for Precision":
                idx = precisions.argmax()
                best_threshold = ths[idx]

            else:  # Optimize F1
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
                idx = f1_scores.argmax()
                best_threshold = ths[idx]

            st.info(f"Using optimized threshold: {round(float(best_threshold), 3)}")

        effective_threshold = float(best_threshold)

        # FINAL PREDICTIONS
        preds = (proba >= effective_threshold).astype(int)

    # Fallback for binary models without predict_proba (rare)
    elif is_binary:
        classes = sorted(y_test.unique())
        y_bin = (y_test == classes[1]).astype(int)
        preds = (preds == classes[1]).astype(int)
        proba = None

    # predict missing targets
    future_preds = None
    if len(X_to_predict) > 0:
        future_preds = pipeline.predict(X_to_predict)

    col1, col2 = st.columns([1, 2])

    if is_binary:

        if not hasattr(pipeline.named_steps["model"], "predict_proba"):
            st.warning("This model does not support probability scores.")
            proba = None

        metrics = {
            "Accuracy": accuracy_score(y_bin, preds),
            "Precision": precision_score(y_bin, preds, zero_division=0),
            "Recall": recall_score(y_bin, preds, zero_division=0),
            "F1": f1_score(y_bin, preds, zero_division=0),
            "ROC AUC": roc_auc_score(y_bin, proba) if proba is not None else "N/A",
            "Effective Threshold": round(float(effective_threshold), 3)
        }

        with col1:
            st.subheader("Metrics")
            st.json(metrics)
            st.info(f"Applied decision threshold: {round(float(effective_threshold), 3)}")

            # ---------- AUC WARNING ----------
            if proba is not None:
                auc_value = metrics["ROC AUC"]
                if auc_value != "N/A" and auc_value < 0.6:
                    st.warning(
                        "⚠️ Model may be unreliable (AUC < 0.6). "
                        "Consider adding features, cleaning data, or trying another model."
                    )

                # ---------- ROC CURVE ----------
                from sklearn.metrics import roc_curve

                fpr, tpr, roc_th = roc_curve(y_bin, proba)

                fig_roc, ax_roc = plt.subplots()

                # ROC curve
                ax_roc.plot(fpr, tpr, label="ROC Curve")
                ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

                # 🔥 SHOW THRESHOLD POINT
                # find point closest to chosen effective threshold
                idx = (np.abs(roc_th - effective_threshold)).argmin()

                ax_roc.scatter(
                    fpr[idx],
                    tpr[idx],
                    color="red",
                    s=80,
                    label=f"Threshold = {round(float(effective_threshold), 3)}"
                )

                ax_roc.set_title("ROC Curve")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()

                st.pyplot(fig_roc)
                plt.close(fig_roc)

            st.subheader("Model Summary")

            if metrics["Recall"] < 0.6:
                st.warning("Low recall — model is missing many positives. Be careful.")
            elif proba is not None and metrics["ROC AUC"] < 0.7:
                st.info("Model is weak. Consider adding more features or data.")
            else:
                st.success("Model looks solid and usable.")

        # recover original class names in correct order
        labels = list(y_test.unique())  # [neg, pos]

        # confusion matrix still works on 0/1…
        cm = confusion_matrix(y_bin, preds, labels=[0, 1])

        # …but we DISPLAY with real labels
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual: {classes[0]}", f"Actual: {classes[1]}"],
            columns=[f"Pred: {classes[0]}", f"Pred: {classes[1]}"]
        )

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df)

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
            plt.close(fig)

    # ======================================
    # FEATURE IMPORTANCE (CORRECTED)
    # ======================================
    st.subheader("Feature Importance")

    try:
        final_model = pipeline.named_steps["model"]

        # 1) Native importance (GBM / RF / Tree models)
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_

            try:
                names = pipeline.named_steps["prep"].get_feature_names_out()
            except:
                names = [f"Feat_{i}" for i in range(len(importances))]

            fi = pd.DataFrame({"Feature": names, "Importance": importances})
            fi = fi.sort_values(by="Importance", ascending=True).tail(20)

            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            ax_imp.barh(fi["Feature"], fi["Importance"], color="#4b72af")
            ax_imp.set_title(f"Native Importance ({model_choice})")
            ax_imp.set_xlabel("Relative Importance")
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        # 2) Permutation Importance (Linear / Logistic / NN)
        else:
            with st.spinner("Calculating Permutation Importance..."):
                result = permutation_importance(
                    pipeline,
                    X_test,
                    y_test,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1
                )

            # Use Raw Columns to prevent Dimension Mismatch
            names = X_test.columns
            imp = result.importances_mean

            fi = pd.DataFrame({"Feature": names, "Importance": imp})
            fi = fi.sort_values(by="Importance", ascending=True)

            # -------------------------------------------------
            # Suggest features to drop (only if toggle enabled)
            # -------------------------------------------------
            if suggest_drops:
                weak = fi[fi["Importance"] <= 0]

                if len(weak) > 0:
                    st.warning(
                        "These features may be hurting the model — consider removing them:\n\n"
                        + ", ".join(list(weak["Feature"]))
                    )
                else:
                    st.success("No harmful features detected 🎯")

            fig_perm, ax_perm = plt.subplots(figsize=(10, 6))
            colors = ["#4caf50" if v > 0 else "#e53935" for v in fi["Importance"]]
            ax_perm.barh(fi["Feature"], fi["Importance"], color=colors)
            ax_perm.set_title("Permutation Importance")
            ax_perm.set_xlabel("Performance Drop if Shuffled")
            st.pyplot(fig_perm)
            plt.close(fig_perm)

    except Exception as e:
        st.error(f"Feature importance could not be calculated: {e}")

    # ------------------ Save Outputs ------------------
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_choice.replace(' ', '_')}.joblib"
    joblib.dump(pipeline, model_path)

    st.success(f"Model saved: {model_path}")

    # test part (trained rows)
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["y_pred"] = preds
    out["Row_Type"] = "train"
    if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
        out["y_proba"] = pipeline.predict_proba(X_test)[:, 1]
        out["Low_Confidence"] = (abs(out["y_proba"] - effective_threshold) <= 0.10)

    # prediction-only rows
    if future_preds is not None:
        future = X_to_predict.copy()
        future["y_true"] = None
        future["y_pred"] = future_preds
        future["Row_Type"] = "predict"

        # add probability for classification
        if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
            future["y_proba"] = pipeline.predict_proba(X_to_predict)[:, 1]
            future["Low_Confidence"] = (abs(future["y_proba"] - effective_threshold) < 0.10)
        out = pd.concat([out, future], axis=0)

    if predict_only:
        out = out[out["Row_Type"] == "predict"]
    st.download_button(
        "Download Predictions CSV",
        out.to_csv(index=False),
        "predictions.csv"
    )
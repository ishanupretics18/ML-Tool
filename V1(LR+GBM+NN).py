import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
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

if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
# Try LightGBM
try:
   import lightgbm as lgb


   HAS_LGB = True
except ImportError:
   HAS_LGB = False


st.set_page_config("ML Ops Tool", layout="wide")
st.title("ML Business Strategy Tool")


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

# ------------------ Column Selection & HYGIENE ------------------
target = st.sidebar.selectbox("Target column", df.columns)

# 1. Clean whitespace and casing
if df[target].dtype == 'object':
    try:
        df[target] = df[target].astype(str).str.strip().str.title()
    except:
        pass

# 2. STANDARDIZE BLANKS (Convert all empty/space/null to actual NaN)
# This ensures we catch every type of "missing" data
df[target] = df[target].replace(r'^\s*$', np.nan, regex=True)
df[target] = df[target].replace(['nan', 'Nan', 'NULL', 'None'], np.nan)

# 3. CRITICAL SPLIT: Train vs Predict
# train_mask finds rows where Target IS NOT EMPTY. These are for Training/Testing.
train_mask = df[target].notna()

# predict_mask finds rows where Target IS EMPTY. These are for Future Prediction.
predict_mask = ~train_mask

# 4. Define the Data Pools
# Raw Training Data (Valid Targets Only)
df_train_all = df[train_mask].copy()

# Raw Prediction Data (Empty Targets Only)
df_predict_all = df[predict_mask].copy()

# Update the display to show the full dataset (so you can see what was uploaded)
predict_only = st.sidebar.toggle("Predict missing targets only", value=False)
st.dataframe(df.head())


model_container = st.sidebar.container()




# --- SMART FEATURE FILTERING ---
# We separate columns into "Safe" (Numeric or Low-Cardinality) and "Risk" (High-Cardinality Text like IDs)
valid_features = []
ignored_features = []


for c in df.columns:
   if c == target:
       continue


   # Check if column is numeric (Always safe)
   if pd.api.types.is_numeric_dtype(df[c]):
       valid_features.append(c)
   else:
       # Check if column is text/categorical
       unique_count = df[c].nunique()
       # If it has fewer than 20 unique values, it's a category (Safe)
       # If it has more, it's likely an ID or Name (Risk)
       if unique_count <= 20:
           valid_features.append(c)
       else:
           ignored_features.append(c)


features = st.sidebar.multiselect(
   "Feature columns",
   options=valid_features + ignored_features,  # Show all, but...
   default=valid_features  # ...only pre-select the safe ones
)


if ignored_features:
   st.sidebar.caption(f"⚠️ Auto-deselected {len(ignored_features)} columns (Too Many Unique Values/IDs).")


if len(features) == 0:
   st.stop()


X = df[features]

y = df[target]

y_raw_train = y[train_mask]

X_train_all = X[train_mask]

if len(X_train_all) < 5:
    st.error(f"❌ Not enough training data! The target '{target}' has {len(X_train_all)} valid rows. Need at least 5.")
    st.stop()
# --- 2. SET UP PREDICTION DATA ---

# These are the rows where Target was NaN

X_to_predict = X[predict_mask]

# --- 3. DETECT PROBLEM TYPE & ENCODE TARGET ---
# We check types ONLY on the valid training data
is_numeric = pd.api.types.is_numeric_dtype(y_raw_train)
unique_count = y_raw_train.nunique()

le = None

# [FIX] CHECK FOR HIGH CARDINALITY TEXT FIRST!
# If it is text AND has > 20 unique values, stop immediately.
if (not is_numeric) and (unique_count > 20):
    st.error(f"⛔ Target Error: '{target}' has {unique_count} unique values. Likely an ID or unique identifier.")
    st.stop()

# [FIX] If we passed the check above, we proceed safely.
# Logic: Classification if text OR (numeric AND few values)
if (not is_numeric) or (is_numeric and unique_count <= 20):
    is_classification = True

    # Import Encoder
    le = LabelEncoder()

    # Fit ONLY on valid training data
    y_encoded = le.fit_transform(y_raw_train.astype(str))

    # Create the final y_train series
    y_train_all = pd.Series(y_encoded, index=y_raw_train.index, name=target)

    # Strictly check if it is binary (2 classes)
    is_binary = (y_train_all.nunique() == 2)

else:
    # Regression Logic
    is_classification = False
    is_binary = False
    y_train_all = y_raw_train  # Use raw numeric values for regression

# A. BINARY TARGET: Calculate Information Value (IV) & Drill Down
if is_classification:
    st.caption("ℹ️ **Information Value (IV):** Ranks features by how well they split 'Yes' vs 'No'.")


    # 1. Define the Math Engine (Returns the Full Table)
    def get_woe_table(df, feature, target):
        lst = []
        temp_df = df.copy()

        # Auto-Binning for Numerics
        if pd.api.types.is_numeric_dtype(temp_df[feature]):
            try:
                # Create 5 bins (Quintiles) for detailed view
                temp_df[feature] = pd.qcut(temp_df[feature], q=5, duplicates='drop').astype(str)
            except:
                pass  # If qcut fails (too few unique values), treat as categorical

        # Calculate Good/Bad counts for each bin
        for val in temp_df[feature].unique():
            lst.append([
                val,
                temp_df[(temp_df[feature] == val) & (temp_df[target] == 1)].shape[0],  # Good (Target=1)
                temp_df[(temp_df[feature] == val) & (temp_df[target] == 0)].shape[0]  # Bad (Target=0)
            ])

        # Build DataFrame
        data = pd.DataFrame(lst, columns=['Value', 'Good', 'Bad'])
        data = data[(data['Good'] > 0) | (data['Bad'] > 0)]  # Remove empty bins

        # Calculate Distributions
        total_good = data['Good'].sum()
        total_bad = data['Bad'].sum()

        # Safety: Add small epsilon to prevent DivideByZero in Log
        data['Dist_Good'] = data['Good'] / total_good
        data['Dist_Bad'] = data['Bad'] / total_bad

        # WoE Formula: ln( %Good / %Bad )
        # If Dist_Bad is 0, we cap WoE to a sensible max/min to prevent crash
        data['WoE'] = np.where(data['Dist_Bad'] == 0, 0,
                               np.log((data['Dist_Good'] + 0.0001) / (data['Dist_Bad'] + 0.0001)))
        data['IV_Contrib'] = (data['Dist_Good'] - data['Dist_Bad']) * data['WoE']

        return data.sort_values(by="WoE", ascending=True)


    # 2. Run IV Loop (Leaderboard)
    iv_data = []


    with st.spinner("Analyzing feature power (IV) on FULL dataset..."):
        # FIX: Only use rows that have a target (ignore prediction rows with NaNs)
        temp_df = df.loc[train_mask].copy()

        minority_class = temp_df[target].value_counts().idxmin()
        temp_df['target_internal'] = (temp_df[target] == minority_class).astype(int)

        for col in features:
            try:
                woe_table = get_woe_table(temp_df, col, 'target_internal')
                iv_score = woe_table['IV_Contrib'].sum()

                # Interpretation
                if iv_score < 0.02:
                    power = "Useless"
                elif iv_score < 0.1:
                    power = "Weak"
                elif iv_score < 0.3:
                    power = "Medium"
                elif iv_score < 0.5:
                    power = "Strong"
                else:
                    power = "Suspicious (Too Good)"

                iv_data.append({"Feature": col, "IV Score": round(iv_score, 4), "Power": power})
            except:
                continue

    # 3. Display Leaderboard
    if iv_data:
        iv_df = pd.DataFrame(iv_data).sort_values("IV Score", ascending=False)


        def highlight_strong(val):
            if val == "Strong" or val == "Suspicious (Too Good)":
                return 'background-color: #d4edda; color: black'
            return ''


        st.dataframe(iv_df.style.applymap(highlight_strong, subset=['Power']), use_container_width=True)

        # --- 4. THE DRILL DOWN SECTION (Select & Inspect) ---
        st.markdown("---")
        st.subheader("🔎 Drill Down: Detailed WoE Patterns")
        st.caption(
            "Inspect any feature to see which specific values drive the risk. (Negative WoE = Risky / Target-Heavy)")

        col_inspect, col_viz = st.columns([1, 2])

        with col_inspect:
            selected_feat = st.selectbox("Select Feature to Inspect:", iv_df["Feature"].tolist())

            # Calculate Detailed Table for Selection
            details = get_woe_table(temp_df, selected_feat, 'target_internal')

            # Show Table
            st.dataframe(
                details[['Value', 'Good', 'Bad', 'WoE']].style.background_gradient(subset=['WoE'], cmap="RdYlGn"),
                use_container_width=True,
                hide_index=True
            )

        with col_viz:
            # Plot WoE Pattern
            fig, ax = plt.subplots(figsize=(8, 4))

            # Color logic: Red for Negative (Risky), Green for Positive (Safe)
            colors = ['#ff4b4b' if x < 0 else '#21c354' for x in details['WoE']]

            ax.barh(details['Value'].astype(str), details['WoE'], color=colors)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_title(f"Weight of Evidence: {selected_feat}")
            ax.set_xlabel("WoE Value (Left = More Likely to be Target)")

            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("Could not calculate IV (likely data format issues).")

# B. REGRESSION TARGET: Calculate Correlation
else:
    st.caption("ℹ️ **Correlation Analysis:** Ranks features by linear relationship with Target.")
    corr_data = []
    numeric_df = X_train_all.select_dtypes(include=np.number)

    if len(numeric_df.columns) > 0:
        # Add target back temporarily for correlation
        numeric_df[target] = y_train_all
        corr_matrix = numeric_df.corr()

        # Extract correlation with target
        if target in corr_matrix.columns:
            target_corr = corr_matrix[target].drop(target)  # Drop self-correlation
            for col, score in target_corr.items():
                if col in features:
                    power = "Weak"
                    if abs(score) > 0.5:
                        power = "Strong"
                    elif abs(score) > 0.3:
                        power = "Medium"

                    corr_data.append({"Feature": col, "Correlation": round(score, 4), "Strength": power})

        corr_df = pd.DataFrame(corr_data).sort_values("Correlation", key=abs, ascending=False)
        st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("Target is numeric, but no numeric features found for correlation.")
# ------------------ Model Selection ------------------
with model_container:
    # UPDATED: Use 'is_classification' instead of 'is_binary'
    if is_classification:
        model_choice = st.selectbox(
            "Model",
            ["Logistic Regression", "GBM", "Neural Network"]
        )
        # Keep threshold stuff for Binary only
        # Keep threshold stuff for Binary only
        if is_binary:
            threshold_mode = st.sidebar.selectbox(
                "Threshold Mode",
                ["Manual", "Optimize for Recall", "Optimize for Precision", "Optimize F1"]
            )
            # This slider must always exist for 'Manual' mode to work
            threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01)
    else:
        model_choice = st.selectbox(
            "Model",
            ["Linear Regression", "GBM", "Neural Network"]
        )

    # --- Enable tuning ---
    enable_tuning = st.sidebar.checkbox(
        "⚡ Enable Hyperparameter Tuning",
        value=False,
        help="If checked, the AI will try random configurations to find the best one."
    )

    # POWER USER FEATURE: Slider appears only if Tuning is ON
    if enable_tuning:
        tuning_iter = st.sidebar.select_slider(
            "Tuning Intensity (Trials)",
            options=[10, 30, 50, 75, 100],  # <--- Specific list of options
            value=10,
            help="Higher values test more combinations but take longer."
        )
    else:
        tuning_iter = 10  # Fallback default

    #Refit Strategy Checkbox (Defined in Sidebar to prevent app reset)
    refit_strategy = st.sidebar.checkbox(
        "🚀 Retrain on 100% data for predictions",
        value=False,
        help="Maximize accuracy by using all available data (Train + Test) for the final CSV."
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

# ------------------ Automated & Manual Imbalance Handling ------------------
if is_classification:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Imbalance Handling")

    actual_ratio = y_train_all.value_counts(normalize=True).min()

    manual_balance = st.sidebar.checkbox("Force Manual Balancing", value=False,
                                         help="Check this to force weights even if data looks balanced.")

    user_threshold = st.sidebar.slider("Auto-balance Threshold (%)", 5, 50, 15) / 100.0

    # Logic: Active if minority ratio is LESS than user threshold
    handle_imbalance = (actual_ratio < user_threshold) or manual_balance

    counts = y_train_all.value_counts()

    # --- DECODE LABELS FOR SIDEBAR TABLE ---
    try:
        # Convert numeric index (0, 1) back to text (No, Yes)
        decoded_labels = le.inverse_transform(counts.index)
        balance_df = pd.DataFrame({
            "Class": decoded_labels,
            "Count": counts.values,
            "Percentage": (y_train_all.value_counts(normalize=True) * 100).round(1).astype(str) + "%"
        })
    except:
        # Fallback if encoder fails
        balance_df = pd.DataFrame({
            "Class": counts.index,
            "Count": counts.values,
            "Percentage": (y_train_all.value_counts(normalize=True) * 100).round(1).astype(str) + "%"
        })

    st.sidebar.dataframe(balance_df, use_container_width=True, hide_index=True)

    if handle_imbalance:
        st.sidebar.success(f"✅ Balancing: ACTIVE {'(Manual)' if manual_balance else '(Auto)'}")
    else:
        st.sidebar.info("ℹ️ Balancing: OFF")
else:
    handle_imbalance = False
X_train, X_test, y_train, y_test = train_test_split(
   X_train_all, y_train_all, test_size=test_size / 100, random_state=42
)

# ------------------ Model Init & Params ------------------
if model_choice == "Linear Regression":
    model = Ridge()
    param_dist = {"model__alpha": np.logspace(-2, 2, 10)}

elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced" if handle_imbalance else None)
    param_dist = {"model__C": np.logspace(-2, 2, 10)}

elif model_choice == "GBM":
    if HAS_LGB:
        if is_binary:
            model = lgb.LGBMClassifier(random_state=42, class_weight="balanced" if handle_imbalance else None)
        else:
            model = lgb.LGBMRegressor(random_state=42)
        param_dist = {
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__n_estimators": [50, 100, 200],
            "model__num_leaves": [20, 31, 50]
        }
    else:
        if is_binary:
            try:
                model = HistGradientBoostingClassifier(class_weight="balanced" if handle_imbalance else None,
                                                       random_state=42)
            except TypeError:
                model = HistGradientBoostingClassifier(random_state=42)
        else:
            model = HistGradientBoostingRegressor(random_state=42)

        param_dist = {
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_iter": [50, 100, 200],
            "model__max_leaf_nodes": [20, 31, 50]
        }

else:  # Neural Network
    if is_binary:
        model = MLPClassifier(max_iter=500, random_state=42)
    else:
        model = MLPRegressor(max_iter=500, random_state=42)

    param_dist = {
        "model__hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate_init": [0.001, 0.01]
    }

pipeline = Pipeline([
   ("prep", preprocessor),
   ("model", model)
])
# ------------------ Train ------------------
if st.button("Train Model"):
    # 1. Train the "Champion" (Default Model)
    with st.spinner("Training Default Model (The Champion)..."):
        pipeline.fit(X_train, y_train)

        # --- FIX: Correct Scoring Logic ---
        # We split Classification vs Regression immediately
        if is_classification:
            y_pred_def = pipeline.predict(X_test)
            score_avg = 'binary' if is_binary else 'weighted'
            baseline_score = f1_score(y_test, y_pred_def, average=score_avg, zero_division=0)
            score_name = "F1 Score"
        else:
            # Regression Mode
            baseline_score = r2_score(y_test, pipeline.predict(X_test))
            score_name = "R2 Score"

    # 2. Run the "Challenger" (Hyperparameter Tuning) - ONLY if checked
    if enable_tuning:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.write(f"⚡ Challenge Round: Preparing to test {tuning_iter} configurations...")

        from sklearn.base import clone

        tuned_pipeline = clone(pipeline)

        # Fix scoring param based on type
        if is_classification:
            tune_metric = 'f1' if is_binary else 'f1_weighted'
        else:
            tune_metric = 'r2'

        search = RandomizedSearchCV(
            tuned_pipeline,
            param_distributions=param_dist,
            n_iter=tuning_iter,
            cv=3,
            random_state=42,
            n_jobs=1,
            scoring=tune_metric
        )

        try:
            import time

            status_text.write(f"⚡ AI is training {tuning_iter} models... This may take a moment.")
            progress_bar.progress(10)

            search.fit(X_train, y_train)

            progress_bar.progress(100)
            status_text.success("✅ Tuning Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            best_model = search.best_estimator_

            # Evaluate Challenger
            if is_classification:
                y_tuned = best_model.predict(X_test)
                score_avg = 'binary' if is_binary else 'weighted'
                tuned_score = f1_score(y_test, y_tuned, average=score_avg, zero_division=0)
            else:
                tuned_score = r2_score(y_test, best_model.predict(X_test))

            # --- THE DECISION ---
            if tuned_score > baseline_score:
                pipeline = best_model
                improvement = (tuned_score - baseline_score)
                st.success(f"🎉 **AI Optimization Successful!**")
                st.markdown(
                    f"The AI beat the default settings. **{score_name} improved by {improvement:.3f}** (from {baseline_score:.3f} to {tuned_score:.3f}).")

                best_params = search.best_params_
                translator = {
                    "model__C": "Strictness (C)", "model__alpha": "Smoothing (Alpha)",
                    "model__learning_rate": "Learning Speed", "model__n_estimators": "Number of Trees",
                    "model__num_leaves": "Tree Complexity", "model__max_depth": "Max Depth",
                    "model__hidden_layer_sizes": "Neural Layers", "model__learning_rate_init": "Init Speed"
                }
                msg = []
                for k, v in best_params.items():
                    name = translator.get(k, k.replace('model__', ''))
                    val_str = f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
                    msg.append(f"**{name}:** {val_str}")
                st.info(f"**Winning Settings:** " + ", ".join(msg))
            else:
                st.info(f"ℹ️ **Optimization Result:** The default model was already excellent.")
                st.markdown(
                    f"The AI tried {tuning_iter} variations but none beat the default {score_name} of **{baseline_score:.3f}**. We kept the safe default model.")

        except MemoryError:
            st.error("⛔ **Server Overload:** The dataset is too large to tune.")
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"⚠️ Tuning skipped due to error: {e}")
            progress_bar.empty()
            status_text.empty()
    else:
        st.success(f"✅ Trained with Standard Settings ({score_name}: {baseline_score:.3f})")

    # 3. Final Predictions
    preds = pipeline.predict(X_test)

    # --- Threshold Logic (Binary Only) ---
    if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
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
            elif threshold_mode == "Optimize for Precision":
                idx = precisions.argmax()
            else:
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
                idx = f1_scores.argmax()
            best_threshold = ths[idx] if idx < len(ths) else 0.5

            st.info(f"Using optimized threshold: {round(float(best_threshold), 3)}")

        effective_threshold = float(best_threshold)
        preds = (proba >= effective_threshold).astype(int)

    elif is_binary:
        classes = sorted(y_test.unique())
        y_bin = (y_test == classes[1]).astype(int)
        preds = (preds == classes[1]).astype(int)
        proba = None

    col1, col2 = st.columns([1, 2])

    # ==========================
    # CLASSIFICATION OUTPUTS
    # ==========================
    if is_classification:
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_prob_full = pipeline.predict_proba(X_test)
            proba = y_prob_full[:, 1] if is_binary else None
        else:
            st.warning("This model does not support probability scores.")
            proba = None

        unique_test = np.unique(y_test)
        unique_pred = np.unique(preds)
        if len(unique_test) <= 2 and len(unique_pred) <= 2:
            avg_method = 'binary'
        else:
            avg_method = 'weighted'

        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average=avg_method, zero_division=0),
            "Recall": recall_score(y_test, preds, average=avg_method, zero_division=0),
            "F1": f1_score(y_test, preds, average=avg_method, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, proba) if (is_binary and proba is not None) else "N/A",
            "Effective Threshold": round(float(effective_threshold), 3) if 'effective_threshold' in locals() else 0.5
        }

        with col1:
            st.subheader("Metrics")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
            m2.metric("F1 Score", f"{metrics['F1']:.3f}")
            m3, m4 = st.columns(2)
            m3.metric("Precision", f"{metrics['Precision']:.1%}")
            m4.metric("Recall", f"{metrics['Recall']:.1%}")

            if is_binary and proba is not None:
                fpr, tpr, roc_th = roc_curve(y_test, proba)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label="ROC Curve")
                ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
                if 'effective_threshold' in locals():
                    idx = (np.abs(roc_th - effective_threshold)).argmin()
                    ax_roc.scatter(fpr[idx], tpr[idx], color="red", s=80, label="Threshold")
                ax_roc.set_title("ROC Curve")
                st.pyplot(fig_roc)
                plt.close(fig_roc)

            st.subheader("Model Summary")
            if metrics["F1"] < 0.6:
                st.warning("Model is struggling (F1 < 0.6). Check your data.")
            else:
                st.success("Model performance is solid.")

            # --- RESTORED: Header ---
            st.subheader("Confusion Matrix")
            # --- FIX: Robust Confusion Matrix with Real Labels ---
            if is_binary:
                # 1. Get the actual string labels for 1 (Pos) and 0 (Neg)
                if 'le' in locals() and le is not None:
                    try:
                        # Inverse transform expects a list/array
                        label_1 = le.inverse_transform([1])[0]
                        label_0 = le.inverse_transform([0])[0]
                    except:
                        label_1, label_0 = "Class 1", "Class 0"
                else:
                    label_1, label_0 = "1", "0"

                # 2. Generate Matrix
                # Note: Sklearn outputs [[TN, FP], [FN, TP]] for labels=[0, 1]
                # We force labels=[1, 0] so it aligns with typical business view (Top-Left = TP)
                cm = confusion_matrix(y_bin, preds, labels=[1, 0])

                # 3. Create DataFrame with proper text tags
                # Transpose (.T) so Predicted is Rows, Actual is Columns (Matches your visual)
                cm_df = pd.DataFrame(cm.T,
                                     index=[f"Pred: {label_1}", f"Pred: {label_0}"],
                                     columns=[f"Actual: {label_1}", f"Actual: {label_0}"])
            else:
                # Multi-class Logic
                if 'le' in locals() and le is not None:
                    known_labels = le.classes_
                    cm = confusion_matrix(y_test, preds, labels=np.arange(len(known_labels)))
                    label_names = [f"{c}" for c in known_labels]
                else:
                    all_labels = sorted(list(set(y_test) | set(preds)))
                    cm = confusion_matrix(y_test, preds, labels=all_labels)
                    label_names = [f"{l}" for l in all_labels]

                cm_df = pd.DataFrame(cm.T,
                                     index=[f"Pred: {x}" for x in label_names],
                                     columns=[f"Actual: {x}" for x in label_names])

            st.dataframe(cm_df)

    # ==========================
    # REGRESSION OUTPUTS (RESTORED)
    # ==========================
    else:
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        n = len(y_test)
        try:
            p = pipeline.named_steps["model"].n_features_in_
        except:
            p = pipeline.named_steps["prep"].transform(X_test).shape[1]

        # Calculate Adjusted R2
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        metrics = {"MAE": mean_absolute_error(y_test, preds), "RMSE": np.sqrt(mse), "R2": r2, "Adj R2": adj_r2}

        # --- RESTORED: DYNAMIC MESSAGES ---
        # A. R-Squared
        if metrics['R2'] > 0.8:
            r2_msg = "🌟 **Excellent.** Explains most variation."
        elif metrics['R2'] > 0.5:
            r2_msg = "✅ **Decent.** Sees main trends."
        else:
            r2_msg = "⚠️ **Poor.** Features don't explain target well."

        # B. Adjusted R-Squared (Bloat Check)
        diff = metrics['R2'] - metrics['Adj R2']
        if metrics['Adj R2'] < 0:
            adj_msg = "⛔ **Critical:** Worse than random guessing."
        elif diff > 0.10:
            adj_msg = f"⚠️ **High Bloat:** Score dropped by {diff:.3f}. Too many useless columns."
        elif diff > 0.05:
            adj_msg = "ℹ️ **Fair:** Moderate penalty applied."
        else:
            adj_msg = "✅ **Efficient:** Model isn't stuffed with junk data."

        # C. MAE Logic
        target_mean = y_test.mean()
        error_pct = (metrics['MAE'] / target_mean) * 100 if target_mean != 0 else 0
        if error_pct < 10:
            mae_msg = f"🌟 **High Precision:** Off by only ~{error_pct:.1f}%."
        elif error_pct < 20:
            mae_msg = f"✅ **Acceptable:** Off by ~{error_pct:.1f}%."
        else:
            mae_msg = f"⚠️ **High Error:** Off by ~{error_pct:.1f}%."

        # D. RMSE Logic
        gap = metrics['RMSE'] - metrics['MAE']
        rmse_msg = "⚠️ **Unstable:** Large outliers detected." if gap > (
                    metrics['MAE'] * 0.5) else "✅ **Stable:** Errors are consistent."

        with col1:
            st.subheader("Metrics")
            # FIX: Use distinct names for the layout columns
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("R² Score", f"{metrics['R2']:.3f}", help=f"{r2_msg}\n(1.0 = Perfect)")
            col_r2.metric("Adj. R²", f"{metrics['Adj R2']:.3f}", help=adj_msg)
            r3, r4 = st.columns(2)
            r3.metric("MAE", f"{metrics['MAE']:.2f}", help=f"**Meaning:**\n{mae_msg}")
            r4.metric("RMSE", f"{metrics['RMSE']:.2f}", help=f"**Stability:**\n{rmse_msg}")

            st.subheader("Model Summary")
            if metrics["R2"] < 0.4:
                st.warning("Very weak model — predictions are unreliable.")
            elif metrics["R2"] < 0.7:
                st.info("Okay model — usable but improve if possible.")
            else:
                st.success("Strong model — predictions are quite reliable.")

        # --- RESTORED: CENTERED RESIDUAL PLOT ---
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
        # BUSINESS INSIGHTS & RELIABILITY CHECK
        # ======================================
        st.markdown("---")
        st.subheader("💼 Business Applicability & Reality Check")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 1️⃣ What question does this answer?")
            if is_binary:
                st.info(f"**\"Is {target} likely to occur?\"**")
                st.markdown(f"It predicts the probability of the **{classes[1]}** class.")
            else:
                st.info(f"**\"What is the expected value of {target}?\"**")
                st.markdown("It estimates the numerical value based on the features provided.")

        with c2:
            st.markdown("#### 2️⃣ What assumptions are made?")

            # --- LOGIC FOR LINEAR MODELS ---
            if model_choice in ["Linear Regression", "Logistic Regression"]:
                st.write("• **Linearity:** Assumes straight-line relationships.")

                # Check Correlation (VIF Proxy)
                numeric_df = X_train.select_dtypes(include=np.number)
                if numeric_df.shape[1] > 1:
                    corr_matrix = numeric_df.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr_pairs = []
                    for col in upper.columns:
                        for row in upper.index:
                            if upper.loc[row, col] > 0.85:
                                high_corr_pairs.append((row, col))

                    if high_corr_pairs:
                        st.warning("⚠️ **Stability Issue (Collinearity):** The following columns duplicate information.")
                        st.markdown("**Actionable Fix (Remove one from each pair):**")
                        for pair in high_corr_pairs[:3]:
                            st.write(
                                f"- **{pair[0]}** is {round(upper.loc[pair[0], pair[1]] * 100)}% identical to **{pair[1]}**.")
                    else:
                        # Explicitly tell the user everything is fine
                        st.success(
                            "✅ **No Action Needed:** No columns need removal. All features provide unique information (Low Collinearity).")
                else:
                    st.success("✅ Data is simple enough.")

            # --- LOGIC FOR GBM ---
            elif model_choice == "GBM":
                st.success("✅ **Flexible:** Handles complex, non-linear patterns automatically.")
                if len(X_train) < 500:
                    st.warning(
                        f"⚠️ **Data Warning:** You only have {len(X_train)} rows. GBMs usually need 1000+ rows to avoid memorizing data (Overfitting).")

            # --- LOGIC FOR NEURAL NETWORKS ---
            else:
                st.write("• **Complexity:** Assumes complex non-linear relationships.")
                if len(X_train) < 1000:
                    st.error(
                        f"⛔ **Data Starvation:** Neural Networks require massive data to learn. You only have {len(X_train)} rows. **Recommendation:** Switch to Linear/Logistic Regression.")
                else:
                    st.info(
                        "ℹ️ **Black Box Warning:** This model is hard to interpret. Use only if accuracy is more important than explaining 'Why' to stakeholders.")

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### 3️⃣ When will it lie to me?")
            liar_list = []

            # Performance checks
            if is_binary:
                if metrics["ROC AUC"] != "N/A" and metrics["ROC AUC"] < 0.65:
                    liar_list.append(
                        "⚠️ **Uncertainty:** The model is guessing often (AUC < 0.65). Don't trust its confidence scores.")
            else:
                if metrics["R2"] < 0.3:
                    liar_list.append("⚠️ **Weak Signal:** The model only explains a tiny part of the variation (<30%).")

            # General checks
            liar_list.append(
                "⚠️ **Data Drift:** If market conditions change (e.g., inflation, new laws), these predictions will fail immediately.")

            if not liar_list:
                st.success("✅ The model is statistically robust on this test data.")
            else:
                for l in liar_list:
                    st.write(l)

        with c4:
            st.markdown("#### 4️⃣ What mistakes will I make?")
            if is_binary:
                # We use the raw confusion matrix 'cm' (Actual=Rows, Pred=Cols)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0

                if fp > fn:
                    st.error(
                        "⚠️ **False Alarms (Type I Error):** The model is 'Trigger Happy'. You will waste resources on people who won't convert.")
                elif fn > fp:
                    st.error(
                        "⚠️ **Missed Opportunities (Type II Error):** The model is 'Too Careful'. You will miss valuable targets.")
                else:
                    st.info("💡 **Balanced:** The model makes False Positives and Negatives at roughly the same rate.")
            else:
                mae = metrics["MAE"]
                st.error(
                    f"⚠️ **Budgeting Error:** Predictions are wrong by **{mae:.2f}** on average. Can your business margin handle this variance?")

        # ======================================
        # ======================================
        # FEATURE IMPORTANCE
        # ======================================
        st.markdown("---")
        st.subheader("⚖️ Feature Importance & Model Diagnostics")

        try:
            final_model = pipeline.named_steps["model"]

            # ======================================================
            # STEP 1️⃣ — PERMUTATION IMPORTANCE (GROUND TRUTH, ALWAYS)
            # ======================================================
            st.markdown("### 🧠 Ground Truth: What Actually Drives Predictions")

            if is_classification:
                perm_scoring = "f1" if is_binary else "f1_weighted"
            else:
                perm_scoring = "r2"

            with st.spinner("Calculating permutation importance (ground truth)..."):
                perm_result = permutation_importance(
                    pipeline,
                    X_test,
                    y_test,
                    n_repeats=10,
                    random_state=42,
                    scoring=perm_scoring,
                    n_jobs=1
                )

            perm_df = pd.DataFrame({
                "Feature": X_test.columns,
                "Perm_Importance": perm_result.importances_mean,
                "Perm_Std": perm_result.importances_std
            })

            perm_df["Stability"] = perm_df["Perm_Importance"] / (perm_df["Perm_Std"] + 1e-9)
            perm_df = perm_df.sort_values("Perm_Importance", ascending=False)

            # ==========================
            # STEP 2️⃣ — STRUCTURAL BELIEF
            # ==========================
            struct_df = None
            struct_type = None

            try:
                raw_names = pipeline.named_steps["prep"].get_feature_names_out()
                clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
            except:
                clean_names = None

            if hasattr(final_model, "feature_importances_"):
                struct_df = pd.DataFrame({
                    "Feature": clean_names if clean_names else [f"Feat_{i}" for i in
                                                                range(len(final_model.feature_importances_))],
                    "Struct_Importance": final_model.feature_importances_
                })
                struct_type = "Tree Split Importance"

            elif hasattr(final_model, "coef_"):
                coef = final_model.coef_[0] if final_model.coef_.ndim > 1 else final_model.coef_
                struct_df = pd.DataFrame({
                    "Feature": clean_names if clean_names else [f"Feat_{i}" for i in range(len(coef))],
                    "Struct_Importance": np.abs(coef)
                })
                struct_type = "Coefficient Magnitude"

            # ======================================================
            # STEP 3️⃣ — DISAGREEMENT & RISK DETECTION ENGINE
            # ======================================================
            st.markdown("### 🚨 Explanation Consistency Check")

            alerts = []

            if struct_df is not None:
                compare_df = perm_df.merge(struct_df, on="Feature", how="left").fillna(0)

                compare_df["Perm_Norm"] = compare_df["Perm_Importance"] / (
                            compare_df["Perm_Importance"].abs().max() + 1e-9)
                compare_df["Struct_Norm"] = compare_df["Struct_Importance"] / (
                            compare_df["Struct_Importance"].abs().max() + 1e-9)

                false_imp = compare_df[
                    (compare_df["Struct_Norm"] > 0.4) &
                    (compare_df["Perm_Norm"] < 0.05)
                    ]

                hidden = compare_df[
                    (compare_df["Struct_Norm"] < 0.05) &
                    (compare_df["Perm_Norm"] > 0.4)
                    ]

                if len(false_imp) > 0:
                    alerts.append(
                        f"⚠️ **False Importance:** {', '.join(false_imp['Feature'].head(3))} "
                        "look important to the model but do not affect real performance."
                    )

                if len(hidden) > 0:
                    alerts.append(
                        f"⚠️ **Hidden Drivers:** {', '.join(hidden['Feature'].head(3))} "
                        "strongly impact predictions despite low model visibility."
                    )

            harmful = perm_df[perm_df["Perm_Importance"] < 0]
            if len(harmful) > 0:
                alerts.append(
                    f"⛔ **Harmful Features:** {', '.join(harmful['Feature'].head(3))} "
                    "actively reduce model quality."
                )

            unstable = perm_df[perm_df["Stability"] < 1]
            if len(unstable) > 0:
                alerts.append(
                    f"⚠️ **Unstable Signals:** {', '.join(unstable['Feature'].head(3))} "
                    "show inconsistent importance. Interpret cautiously."
                )

            if alerts:
                for a in alerts:
                    if a.startswith("⛔"):
                        st.error(a)
                    else:
                        st.warning(a)
            else:
                st.success("✅ Feature explanations are consistent and reliable.")

            # ======================================================
            # STEP 4️⃣ — VISUALIZATION (TRUTH > BELIEF)
            # ======================================================
            st.markdown("### 📊 Top Drivers (Permutation = Ground Truth)")

            plot_df = perm_df.head(20).sort_values("Perm_Importance")

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#e53935" if v < 0 else "#4caf50" for v in plot_df["Perm_Importance"]]
            ax.barh(plot_df["Feature"], plot_df["Perm_Importance"], color=colors)
            ax.set_title("Permutation Importance (Performance Impact)")
            ax.set_xlabel("Performance Change if Feature is Shuffled")
            st.pyplot(fig, clear_figure=True)

            # ======================================================
            # STEP 5️⃣ — EXECUTIVE INTERPRETATION
            # ======================================================
            st.markdown("### 🧑‍💼 Executive Summary")

            st.info(
                "This analysis verifies whether the model is **actually using** the features it claims are important.\n\n"
                "• Green features are **true business drivers**.\n"
                "• Red features actively harm decision quality.\n"
                "• Alerts indicate bias, redundancy, or instability.\n\n"
                "**Action:** Trust only features confirmed by permutation importance."
            )

        except Exception as e:
            st.error(f"Feature importance analysis failed: {e}")

        # ------------------ Save Outputs ------------------
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_choice.replace(' ', '_')}.joblib"
        joblib.dump(pipeline, model_path)

        st.success(f"Model saved: {model_path}")

        # ------------------ Final Refit Strategy & Export ------------------
        st.markdown("---")

        # 1. Prepare "Train" Data
        train_df = X_train.copy()
        train_df["y_true"] = y_train.values
        # [FIX 6] Use np.nan instead of None for dtype safety
        train_df["y_pred"] = np.nan
        train_df["Row_Type"] = "Train"

        # 2. Prepare "Test" Data
        test_df = X_test.copy()
        test_df["y_true"] = y_test.values
        test_df["y_pred"] = preds
        test_df["Row_Type"] = "Test"

        # Handle Probabilities for Test Data
        current_threshold = effective_threshold if 'effective_threshold' in locals() else 0.5

        if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
            test_df["y_proba"] = pipeline.predict_proba(X_test)[:, 1]
            test_df["Low_Confidence"] = (abs(test_df["y_proba"] - current_threshold) <= 0.10)

        # ---------------------------------------------------------
        # 3. FUTURE PREDICTIONS (REFIT STRATEGY)
        # ---------------------------------------------------------
        future_preds = None
        future_proba = None

        if len(X_to_predict) > 0:
            st.subheader("🔮 Predictions on Missing Targets")

            final_pipeline = pipeline

            if refit_strategy:
                with st.spinner("🚀 Retraining on 100% data..."):
                    try:
                        from sklearn.base import clone

                        # [FIX 7] clone() copies parameters automatically, no need for set_params
                        temp_pipeline = clone(pipeline)
                        temp_pipeline.fit(X_train_all, y_train_all)
                        final_pipeline = temp_pipeline
                        st.success("✅ Retraining complete! Predictions are based on 100% of the data.")
                    except MemoryError:
                        st.error("⛔ **Server Out of Memory:** Could not retrain on 100% data.")
                        st.warning("⚠️ **Fallback:** Using the 80% Training model instead.")
                    except Exception as e:
                        st.error(f"⚠️ Retraining failed due to error: {e}")
                        st.info("Using previous model for predictions instead.")
            else:
                st.caption("ℹ️ Using existing model (trained on partial data) for predictions.")

            # --- PREDICT ---
            future_preds = final_pipeline.predict(X_to_predict)

            if is_binary and hasattr(final_pipeline.named_steps["model"], "predict_proba"):
                future_proba = final_pipeline.predict_proba(X_to_predict)[:, 1]
                future_preds = (future_proba >= current_threshold).astype(int)

        # 4. Build Final Combined Dataframe
        final_df = pd.concat([train_df, test_df], axis=0)

        if future_preds is not None:
            future = X_to_predict.copy()
            future["y_true"] = np.nan  # Use nan for consistency
            future["y_pred"] = future_preds
            future["Row_Type"] = "Predict"

            if is_binary and future_proba is not None:
                future["y_proba"] = future_proba
                future["Low_Confidence"] = (abs(future["y_proba"] - current_threshold) < 0.10)

            final_df = pd.concat([final_df, future], axis=0)

        if predict_only:
            final_df = final_df[final_df["Row_Type"] == "Predict"]
            st.info("Displaying/Downloading 'Predict' rows only.")

        # FIX: Decode the predictions back to "Yes/No" if we have an encoder
        if is_classification and 'le' in locals():
            # We check if y_pred is numeric before decoding to avoid errors
            if pd.api.types.is_numeric_dtype(final_df["y_pred"]):
                try:
                    # Convert y_pred back to original strings (e.g., 0 -> "No", 1 -> "Yes")
                    # We use .astype(int) to handle any floats safely
                    final_df["y_pred_label"] = le.inverse_transform(final_df["y_pred"].fillna(0).astype(int))

                    # Optional: Overwrite y_pred or keep both.
                    # Let's keep y_pred as number and add y_pred_label for clarity
                except Exception as e:
                    pass  # Keep as is if decoding fails


    st.write(f"### Final Data ({final_df.shape[0]} rows)")
    # --- SAVE TO MEMORY ---
    st.session_state.final_df = final_df
    st.session_state.model_trained = True

# --- AT THE VERY BOTTOM OF THE SCRIPT (NOT INDENTED) ---
if st.session_state.model_trained and st.session_state.final_df is not None:
    st.markdown("---")
    st.subheader("📥 Export Predictions")

    csv_data = st.session_state.final_df.to_csv(index=False)
    st.download_button(
        label="Download Full Data (Train/Test/Predict)",
        data=csv_data,
        file_name="ml_strategy_results.csv",
        mime="text/csv"
    )
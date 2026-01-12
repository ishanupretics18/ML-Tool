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
from sklearn.model_selection import RandomizedSearchCV

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
predict_only = st.sidebar.toggle("Predict missing targets only", value=False)
st.dataframe(df.head())


model_container = st.sidebar.container()


# ------------------ Column Selection ------------------
target = st.sidebar.selectbox("Target column", df.columns)


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


# Drop rows where target is NaN (REQUIRED for sklearn)
train_mask = y.notna()


X_train_all = X.loc[train_mask]
y_train_all = y.loc[train_mask]

# <---SAFETY CHECK STARTS HERE --->
if len(X_train_all) < 5:
    st.error(
        f"❌ Not enough training data! "
        f"The target column '{target}' only has {len(X_train_all)} valid (non-empty) rows. "
        "You need at least 5 rows of data to train a model."
    )
    st.stop()
# <--- SAFETY CHECK ENDS HERE --->

X_to_predict = X.loc[~train_mask]

# A. BINARY TARGET: Calculate Information Value (IV) & Drill Down
if is_binary:
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

    # REMOVED LIMIT: Now using full dataframe 'df.copy()' instead of 'head(5000)'
    with st.spinner("Analyzing feature power (IV) on FULL dataset..."):
        temp_df = df.copy()

        target_vals = sorted(temp_df[target].unique())
        # Map target: 1 = Minority/Target, 0 = Majority
        temp_df['target_internal'] = (temp_df[target] == target_vals[1]).astype(int)

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

    # --- Enable tuning ---
    enable_tuning = st.sidebar.checkbox(
        "⚡ Enable Hyperparameter Tuning",
        value=False,
        help="If checked, the AI will try random configurations to find the best one."
    )

    # POWER USER FEATURE: Slider appears only if Tuning is ON
    if enable_tuning:
        tuning_iter = st.sidebar.slider(
            "Tuning Intensity",
            min_value=10,
            max_value=50,
            value=10,
            step=10,
            help="10 = Fast (30s). 50 = Thorough (3-5 mins). Higher values test more combinations."
        )
    else:
        tuning_iter = 10  # Fallback default

    #Refit Strategy Checkbox (Defined in Sidebar to prevent app reset)
    refit_strategy = st.sidebar.checkbox(
        "🚀 Retrain on 100% data for predictions",
        value=True,
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
       # Case B: Data is safe -> AUTOMATICALLY OFF (No Toggle shown)
       st.sidebar.success(f"Data is balanced. (Minority: {round(actual_ratio * 100, 1)}%)")
       st.sidebar.info("✅ Balancing: **OFF (Auto-Decided)**")
       handle_imbalance = False


   # --- 2. VISUAL CONFIRMATION ---


   st.sidebar.markdown("#### 📊 Current Class Distribution")


   # Create a clean DataFrame for display
   balance_df = pd.DataFrame({
       "Count": counts,
       "Percentage": (y_train_all.value_counts(normalize=True) * 100).round(1).astype(str) + "%"
   })


   # Show Table
   st.sidebar.dataframe(balance_df, use_container_width=True)


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

        # Calculate Baseline Score (F1 for Binary, R2 for Regression)
        if is_binary:
            y_def = pipeline.predict(X_test)
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                # Use default 0.5 threshold for fair comparison during tuning
                # (We optimize threshold later for the final output)
                proba_def = pipeline.predict_proba(X_test)[:, 1]
                y_def = (proba_def >= 0.5).astype(int)
            baseline_score = f1_score((y_test == sorted(y_test.unique())[1]).astype(int), y_def, zero_division=0)
            score_name = "F1 Score"
        else:
            baseline_score = r2_score(y_test, pipeline.predict(X_test))
            score_name = "R2 Score"

    # 2. Run the "Challenger" (Hyperparameter Tuning) - ONLY if checked
    if enable_tuning:
        with st.spinner(f"⚡ Challenge Round: AI is trying {tuning_iter} random configurations..."):
            # We clone the pipeline so we don't mess up the default one yet
            from sklearn.base import clone

            tuned_pipeline = clone(pipeline)

            search = RandomizedSearchCV(
                tuned_pipeline,
                param_distributions=param_dist,
                n_iter=tuning_iter,  # <--- CONNECTED TO SLIDER
                cv=3,
                random_state=42,
                n_jobs=1,
                # Optimize for the same metric we measure
                scoring='f1' if is_binary else 'r2'
            )

            try:
                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                # Evaluate the Challenger on the SAME test set
                if is_binary:
                    y_tuned = best_model.predict(X_test)
                    if hasattr(best_model.named_steps["model"], "predict_proba"):
                        proba_tuned = best_model.predict_proba(X_test)[:, 1]
                        y_tuned = (proba_tuned >= 0.5).astype(int)
                    tuned_score = f1_score((y_test == sorted(y_test.unique())[1]).astype(int), y_tuned, zero_division=0)
                else:
                    tuned_score = r2_score(y_test, best_model.predict(X_test))

                # --- THE DECISION ---
                if tuned_score > baseline_score:
                    # Case A: AI Won
                    pipeline = best_model  # Replace default with new winner
                    improvement = (tuned_score - baseline_score)

                    # Business Friendly Message
                    st.success(f"🎉 **AI Optimization Successful!**")
                    st.markdown(
                        f"The AI beat the default settings. **{score_name} improved by {improvement:.3f}** (from {baseline_score:.3f} to {tuned_score:.3f}).")

                    # Print Winning Parameters
                    best_params = search.best_params_
                    translator = {
                        "model__C": "Strictness (C)",
                        "model__alpha": "Smoothing (Alpha)",
                        "model__learning_rate": "Learning Speed",
                        "model__n_estimators": "Number of Trees",
                        "model__num_leaves": "Tree Complexity",
                        "model__max_depth": "Max Depth",
                        "model__hidden_layer_sizes": "Neural Layers",
                        "model__learning_rate_init": "Init Speed"
                    }

                    # --- Format the winning settings (With Rounding) ---
                    msg = []
                    for k, v in best_params.items():
                        name = translator.get(k, k.replace('model__', ''))
                        if isinstance(v, (float, np.floating)):
                            val_str = f"{v:.4f}"  # Round to 4 decimals
                        else:
                            val_str = str(v)
                        msg.append(f"**{name}:** {val_str}")

                    st.info(f"**Winning Settings:** " + ", ".join(msg))

                else:
                    # Case B: AI Failed (Default was better)
                    st.info(f"ℹ️ **Optimization Result:** The default model was already excellent.")
                    st.markdown(
                        f"The AI tried {tuning_iter} variations but none beat the default {score_name} of **{baseline_score:.3f}**. We kept the safe default model.")

            except Exception as e:
                st.error(f"⚠️ Tuning skipped due to error: {e}. Using default model.")

    else:
        # Tuning OFF
        st.success(f"✅ Trained with Standard Settings ({score_name}: {baseline_score:.3f})")

    # 3. Final Predictions (Using whichever model won)
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

    # <--- FIXED: This elif is now perfectly aligned with the 'if' above it
    # Fallback for binary models without predict_proba (rare)
    elif is_binary:
        classes = sorted(y_test.unique())
        y_bin = (y_test == classes[1]).astype(int)
        preds = (preds == classes[1]).astype(int)
        proba = None

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
            "Effective Threshold": round(float(effective_threshold), 3) if 'effective_threshold' in locals() else 0.5
        }

        with col1:
            st.subheader("Metrics")

            # --- DYNAMIC INTERPRETATION LOGIC ---

            # 1. AUC Logic
            auc_val = metrics["ROC AUC"]
            if auc_val == "N/A":
                auc_msg = "N/A (Model doesn't support probabilities)"
            elif auc_val > 0.85:
                auc_msg = "🌟 Excellent. The model is very good at distinguishing Yes from No."
            elif auc_val > 0.70:
                auc_msg = "✅ Good. The model is reliable for most predictions."
            elif auc_val > 0.60:
                auc_msg = "⚠️ Fair. The model struggles with hard cases."
            else:
                auc_msg = "⛔ Poor. The model is barely better than a coin flip (Random Guessing)."

            # 2. Precision Logic (Trust)
            prec_val = metrics["Precision"]
            if prec_val > 0.8:
                prec_msg = "High Trust. When it predicts 'Yes', it's usually right."
            elif prec_val < 0.5:
                prec_msg = "⚠️ False Alarm Prone. It predicts 'Yes' too often, leading to wasted effort."
            else:
                prec_msg = "Moderate. Expect some false alarms."

            # 3. Recall Logic (Coverage)
            rec_val = metrics["Recall"]
            if rec_val > 0.8:
                rec_msg = "High Coverage. It finds almost all the 'Yes' cases."
            elif rec_val < 0.5:
                rec_msg = "⚠️ Missed Opportunities. It is missing more than half of the targets."
            else:
                rec_msg = "Moderate. It finds the easy cases but misses harder ones."

            # 4. Accuracy Logic (Context)
            acc_val = metrics["Accuracy"]
            if acc_val > 0.90:
                acc_msg = "🌟 Excellent."
            elif acc_val > 0.80:
                acc_msg = "✅ Good."
            else:
                acc_msg = "⚠️ Fair/Poor."

            # --- DISPLAY METRICS ---
            m1, m2 = st.columns(2)
            m1.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.1%}",
                help=f"**Verdict:** {acc_msg}\n\n**Reality Check:** If your data is imbalanced (e.g. 90% No), high accuracy is meaningless. Check F1 Score instead."
            )
            m2.metric(
                "ROC AUC",
                f"{auc_val:.3f}" if isinstance(auc_val, float) else auc_val,
                help=f"**Prediction Power:**\n{auc_msg}\n\n(1.0 = Perfect, 0.5 = Random)"
            )

            m3, m4 = st.columns(2)
            m3.metric(
                "Precision",
                f"{metrics['Precision']:.1%}",
                help=f"**Trustworthiness:**\n{prec_msg}\n\n(Precision = True Positives / All Predicted Positives)"
            )
            m4.metric(
                "Recall",
                f"{metrics['Recall']:.1%}",
                help=f"**Coverage:**\n{rec_msg}\n\n(Recall = True Positives / All Actual Positives)"
            )

            # 5. F1 Score Logic (Balance)
            f1_val = metrics["F1"]
            if f1_val > 0.8:
                f1_msg = "🌟 Excellent Balance. The model is strong in both Precision and Recall."
            elif f1_val > 0.6:
                f1_msg = "✅ Good. A solid compromise between finding targets and being right."
            elif f1_val > 0.4:
                f1_msg = "⚠️ Fair. The model is struggling to balance false alarms vs. missed targets."
            else:
                f1_msg = "⛔ Poor. The model is failing to identify the positive class effectively."

            m5, m6 = st.columns(2)

            m5.metric(
                "F1 Score",
                f"{metrics['F1']:.3f}",
                help=f"**Harmonic Mean:**\n{f1_msg}\n\n(This is the most important metric if your data is imbalanced, e.g., fraud detection)."
            )

            m6.metric(
                "Threshold",
                f"{metrics['Effective Threshold']:.2f}",
                help="If Probability > This Number, we predict 'Yes'."
            )

            if 'effective_threshold' in locals():
                st.info(f"Applied decision threshold: {round(float(effective_threshold), 3)}")

            # ---------- AUC WARNING ----------
            if proba is not None:
                if auc_val != "N/A" and auc_val < 0.6:
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
                if 'effective_threshold' in locals():
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

            # Calculate CM with "1" (Yes) first, then "0" (No)
        cm = confusion_matrix(y_bin, preds, labels=[1, 0])

        # --- TRANSPOSE TO SWAP AXES ---
        # Now: Rows = Predicted, Columns = Actual
        cm_df = pd.DataFrame(
            cm.T,
            index=[f"Pred: {classes[1]}", f"Pred: {classes[0]}"],  # Yes first
            columns=[f"Actual: {classes[1]}", f"Actual: {classes[0]}"]  # Yes first
        )

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df)


    else:

        # ---------------------------------------------------------

        # 1. CALCULATE RAW METRICS

        # ---------------------------------------------------------

        mse = mean_squared_error(y_test, preds)

        r2 = r2_score(y_test, preds)

        # Calculate Adjusted R2 (Correctly counting features)

        n = len(y_test)

        try:

            # Ask the model how many features it actually used

            p = pipeline.named_steps["model"].n_features_in_

        except AttributeError:

            # Fallback for models that don't track this

            p = pipeline.named_steps["prep"].transform(X_test).shape[1]

        # Prevent Division by Zero

        if n > p + 1:

            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        else:

            adj_r2 = r2

        # --- CRITICAL FIX: Ensure 'metrics' is defined here for later use ---

        metrics = {

            "MAE": mean_absolute_error(y_test, preds),

            "RMSE": np.sqrt(mse),

            "R2": r2,

            "Adj R2": adj_r2

        }

        # ---------------------------------------------------------

        # 2. GENERATE DYNAMIC HELP MESSAGES

        # ---------------------------------------------------------

        # A. R-Squared Logic

        r2_val = metrics['R2']

        if r2_val > 0.8:

            r2_msg = "🌟 **Excellent.** The model explains most of the variation in the target."

        elif r2_val > 0.5:

            r2_msg = "✅ **Decent.** The model sees the main trends, but misses some finer details."

        else:

            r2_msg = "⚠️ **Poor.** The features provided do not explain the target well."

        # B. Adjusted R-Squared Logic

        diff = metrics['R2'] - metrics['Adj R2']

        if metrics['Adj R2'] < 0:

            adj_msg = "⛔ **Critical:** Model is worse than random guessing."

        elif diff > 0.10:

            adj_msg = f"⚠️ **High Bloat:** Score dropped by {diff:.3f}. Too many useless columns."

        elif diff > 0.05:

            adj_msg = "ℹ️ **Fair:** Moderate penalty applied."

        else:

            adj_msg = "✅ **Efficient:** The model is not 'stuffed' with junk data."

        # C. MAE Logic

        target_mean = y_test.mean()

        mae_val = metrics['MAE']

        error_pct = (mae_val / target_mean) * 100 if target_mean != 0 else 0

        if error_pct < 10:

            mae_msg = f"🌟 **High Precision:** Off by only ~{error_pct:.1f}%."

        elif error_pct < 20:

            mae_msg = f"✅ **Acceptable:** Off by ~{error_pct:.1f}%."

        else:

            mae_msg = f"⚠️ **High Error:** Off by ~{error_pct:.1f}%."

        # D. RMSE Logic

        rmse_val = metrics['RMSE']

        gap = rmse_val - mae_val

        if gap > (mae_val * 0.5):

            rmse_msg = "⚠️ **Unstable:** RMSE >> MAE. Occasional massive mistakes (Outliers)."

        else:

            rmse_msg = "✅ **Stable:** RMSE is close to MAE."

        # ---------------------------------------------------------

        # 3. DISPLAY METRICS (FIXED: 2x2 GRID)

        # ---------------------------------------------------------

        with col1:

            st.subheader("Metrics")

            # Row 1: R2 and Adj R2

            r1c1, r1c2 = st.columns(2)

            r1c1.metric("R² Score", f"{metrics['R2']:.3f}", help=f"{r2_msg}\n(1.0 = Perfect)")

            r1c2.metric("Adj. R²", f"{metrics['Adj R2']:.3f}", help=adj_msg)

            # Row 2: MAE and RMSE

            r2c1, r2c2 = st.columns(2)

            r2c1.metric("MAE", f"{metrics['MAE']:.2f}", help=f"**Meaning:**\n{mae_msg}")

            r2c2.metric("RMSE", f"{metrics['RMSE']:.2f}", help=f"**Stability:**\n{rmse_msg}")

            st.subheader("Model Summary")

            if metrics["R2"] < 0.4:

                st.warning("Very weak model — predictions are unreliable.")

            elif metrics["R2"] < 0.7:

                st.info("Okay model — usable but improve if possible.")

            else:

                st.success("Strong model — predictions are quite reliable.")

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
                tn, fp, fn, tp = cm.ravel()

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
        # FEATURE IMPORTANCE (UNIFIED & AUTOMATED)
        # ======================================
        st.markdown("---")
        st.subheader("Feature Importance")

        try:
            final_model = pipeline.named_steps["model"]

            # ------------------------------------------------------
            # 1) Native Importance (GBM / Random Forest / Decision Trees)
            # ------------------------------------------------------
            if hasattr(final_model, "feature_importances_"):
                importances = final_model.feature_importances_

                try:
                    names = pipeline.named_steps["prep"].get_feature_names_out()
                except:
                    names = [f"Feat_{i}" for i in range(len(importances))]

                fi = pd.DataFrame({"Feature": names, "Importance": importances})

                # --- AUTO-SUGGEST FOR GBM (Check for Zeros) ---
                useless = fi[fi["Importance"] == 0]

                if len(useless) > 0:
                    # Check which original columns caused this explosion
                    high_card_culprits = []
                    for c in cat_cols:
                        if c in df.columns:  # Safety check
                            unique_count = df[c].nunique()
                            if unique_count > 20:
                                high_card_culprits.append(f"{c} ({unique_count} features)")

                    warning_msg = (
                        f"⚠️ **Optimization Tip:** Found **{len(useless)} features** with 0.0 importance (Useless).\n"
                    )

                    # --- NEW: Explicitly list the feature names ---
                    warning_msg += f"\n**Features to Remove:** {', '.join(useless['Feature'].tolist())}\n\n"

                    if high_card_culprits:
                        warning_msg += "**Likely Cause (High Cardinality columns):**\n- "
                        warning_msg += "\n- ".join(high_card_culprits)
                    else:
                        warning_msg += "These features simply didn't help the model learn anything. You can safely drop them."

                    st.warning(warning_msg)

                else:
                    st.success("✅ All features are contributing! No completely useless features found.")
                # Plot top 20
                fi = fi.sort_values(by="Importance", ascending=True).tail(20)

                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                ax_imp.barh(fi["Feature"], fi["Importance"], color="#4b72af")
                ax_imp.set_title(f"Native Importance ({model_choice})")
                ax_imp.set_xlabel("Relative Importance (Gain)")
                st.pyplot(fig_imp)
                plt.close(fig_imp)
            # ------------------------------------------------------
            # 2) Permutation Importance (Linear / Logistic / NN)
            # ------------------------------------------------------
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

                names = X_test.columns
                imp = result.importances_mean

                fi = pd.DataFrame({"Feature": names, "Importance": imp})
                fi = fi.sort_values(by="Importance", ascending=True)

                # --- AUTO-SUGGEST FOR PERMUTATION (Check for Negatives) ---
                weak = fi[fi["Importance"] <= 0]
                if len(weak) > 0:
                    st.warning(
                        "⚠️ **Optimization Tip:** These features seem to be hurting accuracy (Importance ≤ 0).\n"
                        "Consider removing them to improve the model:\n\n"
                        f"**{', '.join(list(weak['Feature']))}**"
                    )
                else:
                    st.success("✅ All features are contributing positively! No drops needed.")

                fig_perm, ax_perm = plt.subplots(figsize=(10, 6))
                colors = ["#e53935" if v <= 0 else "#4caf50" for v in fi["Importance"]]

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

        # ------------------ Final Refit Strategy & Export ------------------
        st.markdown("---")

        # 1. Prepare "Train" Data (For reference)
        train_df = X_train.copy()
        train_df["y_true"] = y_train.values
        train_df["y_pred"] = None  # We don't usually predict on train, or you can add pipeline.predict(X_train) here
        train_df["Row_Type"] = "Train"

        # 2. Prepare "Test" Data (The 20% validation set)
        test_df = X_test.copy()
        test_df["y_true"] = y_test.values
        test_df["y_pred"] = preds
        test_df["Row_Type"] = "Test"

        if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
            test_df["y_proba"] = pipeline.predict_proba(X_test)[:, 1]
            test_df["Low_Confidence"] = (abs(test_df["y_proba"] - effective_threshold) <= 0.10)

        # 3. Future Predictions (Refit Logic)
        future_preds = None
        future_proba = None

        # Only show this if there is data to predict
        if len(X_to_predict) > 0:
            st.subheader("🔮 Predictions")

            if refit_strategy:
                with st.spinner("Retraining on full dataset..."):
                    try:
                        pipeline.fit(X_train_all, y_train_all)
                        future_preds = pipeline.predict(X_to_predict)

                        if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
                            future_proba = pipeline.predict_proba(X_to_predict)[:, 1]
                            future_preds = (future_proba >= effective_threshold).astype(int)
                    except Exception as e:
                        st.error(f"Retraining failed: {e}")
            else:
                # Use existing 80% model
                future_preds = pipeline.predict(X_to_predict)
                if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
                    future_proba = pipeline.predict_proba(X_to_predict)[:, 1]
                    future_preds = (future_proba >= effective_threshold).astype(int)

        # 4. Build Final Combined Dataframe
        # Start with Train + Test
        final_df = pd.concat([train_df, test_df], axis=0)

        # Add Predictions if they exist
        if future_preds is not None:
            future = X_to_predict.copy()
            future["y_true"] = None
            future["y_pred"] = future_preds
            future["Row_Type"] = "Predict"  # Labeled as Predict

            if future_proba is not None:
                future["y_proba"] = future_proba
                future["Low_Confidence"] = (abs(future["y_proba"] - effective_threshold) < 0.10)

            final_df = pd.concat([final_df, future], axis=0)

        # Filter if user requested "Predict only"
        if predict_only:
            final_df = final_df[final_df["Row_Type"] == "Predict"]

        st.download_button(
            "Download Full Data (Train/Test/Predict)",
            final_df.to_csv(index=False),
            "predictions.csv"
        )
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.linear_model import (
   Ridge, LogisticRegression
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier


from sklearn.metrics import (
   mean_absolute_error, mean_squared_error, r2_score,
   accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
   confusion_matrix
)

# ==============================
# 🏭 INDUSTRY PRESET REGISTRY
# ==============================

LINEAR_PRESETS = {
    "Low Regularization (Stable Data)": {
        "model__alpha": 0.1
    },
    "Balanced (Industry Default)": {
        "model__alpha": 1.0
    },
    "High Regularization (Noisy / Many Features)": {
        "model__alpha": 10.0
    }
}

LOGISTIC_PRESETS = {
    "High Precision (Conservative)": {
        "model__C": 0.3,
        "model__solver": "liblinear",
        "model__max_iter": 1000
    },
    "Balanced (Industry Default)": {
        "model__C": 1.0,
        "model__solver": "lbfgs",
        "model__max_iter": 1000

    },
    "High Recall (Aggressive)": {
        "model__C": 3.0,
        "model__solver": "lbfgs",
        "model__max_iter": 1000
    }
}

GBM_PRESETS = {
    "Fast & Safe (Low Overfitting)": {
        "model__learning_rate": 0.05,
        "model__n_estimators": 100,
        "model__num_leaves": 31
    },
    "Balanced Production (Industry Default)": {
        "model__learning_rate": 0.1,
        "model__n_estimators": 200,
        "model__num_leaves": 31
    },
    "High Accuracy (Large Data Only)": {
        "model__learning_rate": 0.03,
        "model__n_estimators": 500,
        "model__num_leaves": 50
    }
}

NN_PRESETS = {
    "Small Data Safe": {
        "model__hidden_layer_sizes": (50,),
        "model__alpha": 0.01,
        "model__learning_rate_init": 0.001
    },
    "Balanced (Industry Default)": {
        "model__hidden_layer_sizes": (100, 50),
        "model__alpha": 0.001,
        "model__learning_rate_init": 0.001
    },
    "High Capacity (Large Data Only)": {
        "model__hidden_layer_sizes": (150, 100, 50),
        "model__alpha": 0.0005,
        "model__learning_rate_init": 0.0005
    }
}

INDUSTRY_PRESETS = {
    "Linear Regression": LINEAR_PRESETS,
    "Logistic Regression": LOGISTIC_PRESETS,
    "GBM": GBM_PRESETS,
    "Neural Network": NN_PRESETS
}




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
    compare_models = st.sidebar.checkbox(
        "🔍 Compare all selected models (Recommended)",
        value=True,
        help="If enabled, all selected models are evaluated and the best one is automatically selected."
    )

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
    st.sidebar.markdown("---")

    st.sidebar.header("⚙️ Optimization Strategy")

    use_custom_params = st.sidebar.checkbox(
        "🛠️ Manually set model hyperparameters",
        value=False,
        help="Use this only if you know what these parameters do. Overrides defaults."
    )

    custom_params = {}

    if use_custom_params:
        st.sidebar.markdown("⚙️ Custom Hyperparameters")

        if model_choice == "Logistic Regression":
            custom_params["model__C"] = st.sidebar.number_input(
                "C (Regularization Strength)", 0.001, 100.0, 1.0
            )
            custom_params["model__max_iter"] = st.sidebar.number_input(
                "Max Iterations", 100, 5000, 1000
            )

        elif model_choice == "Linear Regression":
            custom_params["model__alpha"] = st.sidebar.number_input(
                "Alpha (Regularization)", 0.001, 100.0, 1.0
            )


        elif model_choice == "GBM":

            custom_params["model__learning_rate"] = st.sidebar.number_input(

                "Learning Rate", 0.001, 0.5, 0.1

            )

            if HAS_LGB:

                custom_params["model__n_estimators"] = st.sidebar.number_input(

                    "Estimators", 50, 1000, 200

                )

            else:

                custom_params["model__max_iter"] = st.sidebar.number_input(

                    "Iterations", 50, 1000, 200

                )


        elif model_choice == "Neural Network":
            custom_params["model__alpha"] = st.sidebar.number_input(
                "L2 Alpha", 0.00001, 0.1, 0.001
            )
            custom_params["model__learning_rate_init"] = st.sidebar.number_input(
                "Learning Rate", 0.0001, 0.1, 0.001
            )

    if use_custom_params:
        custom_mode = st.sidebar.radio(
            "Final Model Rule",
            [
                "Best model wins (Recommended)",
                "Always use my custom model"
            ]
        )
    else:
        custom_mode = "Best model wins (Recommended)"



    use_presets = st.sidebar.checkbox(
        "🏭 Use Industry Presets",
        value=False,
        help="Evaluate proven industry-standard configurations"
    )

    if use_presets:
        preset_name = st.sidebar.selectbox(
            "Preset Strategy",
            list(INDUSTRY_PRESETS[model_choice].keys())
        )

    st.sidebar.subheader("🎛️ Smart Hyperparameter Exploration")

    # --- Enable tuning ---
    enable_tuning = st.sidebar.checkbox(
        "🎛️ Smart Hyperparameter Tuning",
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

st.sidebar.markdown("---")
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
    model_scores = {}
    model_objects = {}

    # ==============================
    # CUSTOM HYPERPARAMETER MODEL
    # ==============================
    from sklearn.base import clone

    if use_custom_params and custom_params:
        st.markdown("### 🛠️ Evaluating Custom Hyperparameter Model")

        try:
            custom_pipeline = clone(pipeline)
            custom_pipeline.set_params(**custom_params)
            custom_pipeline.fit(X_train, y_train)


            if is_classification:
                preds_custom = custom_pipeline.predict(X_test)
                avg = 'binary' if is_binary else 'weighted'
                custom_score = f1_score(y_test, preds_custom, average=avg, zero_division=0)
            else:
                custom_score = r2_score(y_test, custom_pipeline.predict(X_test))

            model_scores["Custom (User Defined)"] = custom_score
            model_objects["Custom (User Defined)"] = custom_pipeline

            st.success(f"🛠️ Custom Model Score: {round(custom_score, 4)}")

        except Exception as e:
            st.error(f"Custom hyperparameter model failed: {e}")

    # 1. Train the "Champion" (Default Model)
    with st.spinner("Training Default Model (The Champion)..."):
        pipeline.fit(X_train, y_train)
        score_name = "F1 Score" if is_classification else "R2 Score"

        # --- FIX: Correct Scoring Logic ---
        # We split Classification vs Regression immediately
        if is_classification:
            y_pred_def = pipeline.predict(X_test)
            score_avg = 'binary' if is_binary else 'weighted'
            baseline_score = f1_score(y_test, y_pred_def, average=score_avg, zero_division=0)
        else:
            baseline_score = r2_score(y_test, pipeline.predict(X_test))

    # ✅ Register Default Model
    model_scores["Default"] = baseline_score
    model_objects["Default"] = pipeline

    # ==============================
    # 🏭 INDUSTRY PRESET MODELS
    # ==============================
    if use_presets:
        st.markdown("### 🏭 Evaluating Industry Presets")

        preset_params = INDUSTRY_PRESETS[model_choice][preset_name]

        try:
            preset_pipeline = clone(pipeline)
            safe_params = preset_params.copy()

            # Remove unsupported params for HistGB
            # Remove unsupported params for HistGradientBoosting
            if not HAS_LGB:
                safe_params.pop("model__num_leaves", None)
                safe_params.pop("model__n_estimators", None)

            preset_pipeline.set_params(**safe_params)

            preset_pipeline.fit(X_train, y_train)

            if is_classification:
                preds_preset = preset_pipeline.predict(X_test)
                avg = 'binary' if is_binary else 'weighted'
                preset_score = f1_score(y_test, preds_preset, average=avg, zero_division=0)
            else:
                preset_score = r2_score(y_test, preset_pipeline.predict(X_test))

            model_scores[f"Preset: {preset_name}"] = preset_score
            model_objects[f"Preset: {preset_name}"] = preset_pipeline

            st.success(f"🏭 Preset '{preset_name}' Score: {round(preset_score, 4)}")

        except Exception as e:
            st.error(f"Preset evaluation failed: {e}")

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

            # ✅ Register RandomSearch candidate
            model_scores["RandomSearch"] = tuned_score
            model_objects["RandomSearch"] = best_model


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

    if compare_models:
        st.subheader("📊 Model Comparison")
        st.dataframe(
            pd.DataFrame.from_dict(model_scores, orient="index", columns=["Score"])
            .sort_values("Score", ascending=False)
        )

    # ==============================
    # 🏆 FINAL MODEL SELECTION (AUTHORITATIVE)
    # ==============================

    winner_name = None
    winner_reason = None

    if compare_models:
        # User explicitly forces custom model
        if (
                use_custom_params
                and custom_mode == "Always use my custom model"
                and "Custom (User Defined)" in model_objects
        ):
            winner_name = "Custom (User Defined)"
            winner_reason = "User-forced custom hyperparameters override all comparisons."

        else:
            # Pick best scoring model
            winner_name = max(model_scores, key=model_scores.get)
            winner_reason = "Selected highest performing model across all evaluated candidates."

    else:
        # No comparison → safe defaults
        if (
                use_custom_params
                and custom_mode == "Always use my custom model"
                and "Custom (User Defined)" in model_objects
        ):
            winner_name = "Custom (User Defined)"
            winner_reason = "User forced custom model without comparison."
        else:
            winner_name = "Default"
            winner_reason = "Comparison disabled. Default safe model selected."

    # 🔒 FINAL ASSIGNMENT (ONLY PLACE WHERE PIPELINE IS SET)
    pipeline = model_objects[winner_name]

    # ==============================
    # 📢 FINAL USER COMMUNICATION
    # ==============================

    st.success(f"🏆 Final Model Selected: **{winner_name}**")
    st.caption(f"Reason: {winner_reason}")

    # 3. Final Predictions
    preds = pipeline.predict(X_test)
    effective_threshold = 0.5

    # --- Threshold Logic (Binary Only) ---
    if is_binary and hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_bin = y_test.astype(int)
        proba = pipeline.predict_proba(X_test)[:, 1]

        from sklearn.metrics import precision_recall_curve

        if threshold_mode == "Manual":
            best_threshold = threshold
        else:
            precisions, recalls, ths = precision_recall_curve(y_bin, proba)
            valid_len = len(ths)

            if threshold_mode == "Optimize for Recall":
                idx = np.argmax(recalls[:valid_len])
            elif threshold_mode == "Optimize for Precision":
                idx = np.argmax(precisions[:valid_len])
            else:
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
                idx = np.argmax(f1_scores[:valid_len])

            best_threshold = float(ths[idx]) if valid_len > 0 else 0.5

            st.info(f"Using optimized threshold: {round(float(best_threshold), 3)}")

        effective_threshold = float(best_threshold)
        preds = (proba >= effective_threshold).astype(int)


    elif is_binary:

        y_bin = y_test.astype(int)

        preds = preds.astype(int)

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
            "ROC AUC": (
                roc_auc_score(y_test, proba)
                if is_binary and proba is not None and len(np.unique(y_test)) > 1
                else "N/A"
            ),

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

            cm = None

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
            positive_label = le.inverse_transform([1])[0] if le is not None else "Positive Class"
            st.markdown(f"It predicts the probability of the **{positive_label}** class.")
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

        # --- Classification logic ---
        if is_classification:
            if (
                    is_binary
                    and metrics.get("ROC AUC") != "N/A"
                    and metrics.get("ROC AUC", 1) < 0.65
            ):
                liar_list.append(
                    "⚠️ **Uncertainty:** The model is guessing often (AUC < 0.65). "
                    "Do not trust probability scores."
                )

        # --- Regression logic ---
        else:
            if metrics.get("R2", 1) < 0.3:
                liar_list.append(
                    "⚠️ **Weak Signal:** The model explains very little variation (<30%)."
                )

        # General warning (always applicable)
        liar_list.append(
            "⚠️ **Data Drift Risk:** If real-world conditions change, predictions may fail."
        )

        if not liar_list:
            st.success("✅ The model is statistically robust on this test data.")
        else:
            for l in liar_list:
                st.write(l)

    with c4:
        st.markdown("#### 4️⃣ What mistakes will I make?")
        if is_binary and 'cm' in locals() and cm is not None and 'cm_df' in locals():
            # Confusion matrix computed in sklearn format, then transposed for business-friendly display
            if cm.shape == (2, 2):
                # cm is transposed with labels=[1,0]
                tp = cm_df.loc[f"Pred: {label_1}", f"Actual: {label_1}"]
                fp = cm_df.loc[f"Pred: {label_1}", f"Actual: {label_0}"]
                fn = cm_df.loc[f"Pred: {label_0}", f"Actual: {label_1}"]
                tn = cm_df.loc[f"Pred: {label_0}", f"Actual: {label_0}"]
            else:
                tn = fp = fn = tp = 0

            if fp > fn:
                st.error(
                    "⚠️ **False Alarms (Type I Error):** The model is 'Trigger Happy'. "
                    "You will waste resources on people who won't convert."
                )
            elif fn > fp:
                st.error(
                    "⚠️ **Missed Opportunities (Type II Error):** The model is 'Too Careful'. "
                    "You will miss valuable targets."
                )
            else:
                st.info("💡 **Balanced:** The model makes False Positives and Negatives at roughly the same rate.")
        else:
            st.info("ℹ️ Confusion-based risk analysis is not applicable for this model.")

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

        # ===============================
        # 🔧 AUTHORITATIVE FEATURE NAMES (NO GUESSING)
        # ===============================
        perm_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Perm_Importance": perm_result.importances_mean,
            "Perm_Std": perm_result.importances_std
        })

        perm_df["Stability"] = perm_df["Perm_Importance"] / (perm_df["Perm_Std"] + 1e-9)
        perm_df = perm_df.sort_values("Perm_Importance", ascending=False)

        # ==========================
        # ==========================
        # STEP 2️⃣ — MODEL INTERNALS (ADVANCED)
        # ==========================
        struct_df = None
        struct_type = None

        with st.expander("🔬 Advanced Diagnostics: Model Internal Importance"):
            st.warning(
                "⚠️ **Diagnostic Only — Do NOT use for business decisions.**\n\n"
                "This shows how the model internally used features (splits / coefficients).\n"
                "Models often overweight correlated or high-cardinality features.\n\n"
                "**Ground truth remains Permutation Importance above.**"
            )

            try:
                raw_names = pipeline.named_steps["prep"].get_feature_names_out()
                clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
            except:
                clean_names = None

            if hasattr(final_model, "feature_importances_"):
                struct_df = pd.DataFrame({
                    "Feature": clean_names if clean_names else
                    [f"Feat_{i}" for i in range(len(final_model.feature_importances_))],
                    "Struct_Importance": final_model.feature_importances_
                })
                struct_type = "Tree Split Importance"

            elif hasattr(final_model, "coef_"):
                coef = final_model.coef_[0] if final_model.coef_.ndim > 1 else final_model.coef_
                struct_df = pd.DataFrame({
                    "Feature": clean_names if clean_names else
                    [f"Feat_{i}" for i in range(len(coef))],
                    "Struct_Importance": np.abs(coef)
                })
                struct_type = "Coefficient Magnitude"

            if struct_df is not None:
                st.caption(f"Internal method used: **{struct_type}**")
                st.dataframe(
                    struct_df.sort_values("Struct_Importance", ascending=False).head(20),
                    use_container_width=True
                )
            else:
                st.info("This model does not expose native feature importance.")

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
                    "strongly impact predictions."
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

        plot_df = perm_df.sort_values("Perm_Importance")

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
            "• Near-zero features have little or no impact.\n"
            "• Red features (rare) indicate harmful signals or data leakage.\n"
            "• Alerts indicate bias, redundancy, or instability.\n\n"
            "**Action:** Trust only features confirmed by permutation importance."
        )


    except Exception as e:
        st.error(f"Feature importance analysis failed: {e}")

    # ------------------ Save Outputs ------------------
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{winner_name.replace(' ', '_')}.joblib"
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


# FIX: Decode predictions back to original labels (SAFE)

# ===========================
# ===========================
# FINAL GUARANTEE (CRITICAL)
# ===========================
if "final_df" in locals():
    st.session_state.final_df = final_df
    st.session_state.model_trained = True

# ===========================
# LABEL DECODING (FINAL SAFE)
# ===========================
if (
    is_classification
    and le is not None
    and st.session_state.get("final_df") is not None
    and not st.session_state.final_df.empty
    and "y_pred" in st.session_state.final_df.columns
):
    try:
        valid_mask = st.session_state.final_df["y_pred"].notna()
        st.session_state.final_df.loc[valid_mask, "y_pred_label"] = (
            le.inverse_transform(
                st.session_state.final_df.loc[valid_mask, "y_pred"].astype(int)
            )
        )
    except Exception as e:
        st.warning(f"Label decoding skipped: {e}")


    # SAFE DISPLAY (uses session_state)
    if st.session_state.get("final_df") is not None:
        st.write(f"### Final Data ({st.session_state.final_df.shape[0]} rows)")

# --- AT THE VERY BOTTOM OF THE SCRIPT (NOT INDENTED) ---
if st.session_state.get('model_trained') and st.session_state.get('final_df') is not None:
    st.markdown("---")
    st.subheader("📥 Export Predictions")

    csv_data = st.session_state.final_df.to_csv(index=False)
    st.download_button(
        label="Download Full Data (Train/Test/Predict)",
        data=csv_data,
        file_name="ml_strategy_results.csv",
        mime="text/csv"
    )
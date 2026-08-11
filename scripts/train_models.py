import warnings
from pathlib import Path
import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

warnings.filterwarnings("ignore")

# ==========================================

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent.parent
else:
    current_dir = Path.cwd()
    BASE_DIR = current_dir.parent if current_dir.name == "scripts" else current_dir

# Set target paths
# PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "split_data.pkl"
# MODEL_DIR = BASE_DIR / "models"
# MODEL_DIR.mkdir(parents=True, exist_ok=True)
# Set target paths (Updated to point to 'artifacts' folder)
BASE_DIR = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
PROCESSED_DATA_PATH = BASE_DIR / "artifacts" / "split_data.pkl"
MODEL_DIR = BASE_DIR / "artifacts"  # Or keep as "models" if your Streamlit app checks there
MODEL_DIR.mkdir(parents=True, exist_ok=True)


print(f"Project Root: {BASE_DIR}")
print(f"Loading preprocessed data from: {PROCESSED_DATA_PATH}")
print(f"Saving models to: {MODEL_DIR}")

# Load the split data
saved_data = joblib.load(PROCESSED_DATA_PATH)

# ==========================================

X_train = saved_data["X_train"]
y_train = saved_data["y_train"]
X_val = saved_data["X_val"]
y_val = saved_data["y_val"]

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight_value = neg_count / pos_count

# ==========================================



# ==========================================

# ------------------------------------------------------------------------------
# Step 2: Define Models and Explicit Output Filenames
# ------------------------------------------------------------------------------
# Key: Display Name | Value: (Model Object, Respective Saved Filename)
models_to_train = {
    "Baseline Dummy": (
        DummyClassifier(strategy="most_frequent"),
        "baseline_dummy.pkl",
    ),
    "Baseline Logistic Regression": (
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "baseline_logistic_regression.pkl",
    ),
    "Baseline Decision Tree": (
        DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "baseline_decision_tree.pkl",
    ),
    "Advanced Random Forest": (
        RandomForestClassifier(
            n_estimators=150, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "advanced_random_forest.pkl",
    ),
}

if XGBClassifier:
    models_to_train["Advanced XGBoost"] = (
        XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight_value,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "advanced_xgboost.pkl",
    )

if LGBMClassifier:
    models_to_train["Advanced LightGBM"] = (
        LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight_value,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        "advanced_lightgbm.pkl",
    )

if CatBoostClassifier:
    models_to_train["Advanced CatBoost"] = (
        CatBoostClassifier(
            iterations=150,
            depth=5,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        ),
        "advanced_catboost.pkl",
    )

# ==========================================

# ------------------------------------------------------------------------------
# Step 3: Train, Evaluate, and Save Each Model
# ------------------------------------------------------------------------------
print("\n--- Step 2: Training & Saving Models with Respective Names ---")

results = []
trained_models = {}

for name, (model, filename) in models_to_train.items():
    print(f"\n⏳ Training: {name}...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = y_pred

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_proba)

    # SAVE MODEL WITH RESPECTIVE FILENAME
    save_path = MODEL_DIR / filename
    joblib.dump(model, save_path)
    print(f"   💾 Saved model artifact to: {save_path}")

    results.append({
        "Model Name": name,
        "Filename": filename,
        "ROC-AUC": round(auc, 4),
        "F1-Score": round(f1, 4),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "Accuracy": round(acc, 4),
    })

    trained_models[name] = (model, filename)

# ==========================================

# ------------------------------------------------------------------------------
# Step 4: Summary & Champion Selection
# ------------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

print("\n==========================================================================")
print("                       SAVED MODELS & METRICS                             ")
print("==========================================================================")
print(results_df[["Model Name", "Filename", "ROC-AUC", "F1-Score", "Accuracy"]].to_string(index=False))

# Export full evaluation table
results_df.to_csv(MODEL_DIR / "model_benchmark_results.csv", index=False)

# ==========================================

# Copy champion model to best_model.pkl
champion_row = results_df.iloc[0]
champion_name = champion_row["Model Name"]
champion_model, champion_filename = trained_models[champion_name]

joblib.dump(champion_model, MODEL_DIR / "best_model.pkl")

print("\n==========================================================================")
print(f"🏆 CHAMPION MODEL: {champion_name}")
print(f"📁 Respective File: {MODEL_DIR / champion_filename}")
print(f"⭐ Copied to:     {MODEL_DIR / 'best_model.pkl'}")
print("==========================================================================")

# ==========================================



# ==========================================



# ==========================================



# ==========================================



# ==========================================


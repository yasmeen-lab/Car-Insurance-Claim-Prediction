import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================

# Step 1: Define folder paths
BASE_DIR = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"

print(f"Loading data from: {TRAIN_PATH}")
print("--- Step 1: Loading Raw Data ---")
df = pd.read_csv(TRAIN_PATH)

# Let's inspect the data shape and first few rows
print(f"Total rows and columns: {df.shape}")
print(df.head(3))

# ==========================================

# Step 2: Drop unused ID columns if present
if "policy_id" in df.columns:
    df = df.drop(columns=["policy_id"])

# Step 3: Extract numerical numbers from string columns (max_torque & max_power)
print("\n--- Step 2: Cleaning Torque and Power Features ---")

# ==========================================

def extract_torque_value(text):
    """Helper function to extract torque (Nm) from strings like '113Nm@4400rpm'."""
    match = re.search(r"([\d\.]+)Nm", str(text))
    return float(match.group(1)) if match else np.nan


def extract_power_value(text):
    """Helper function to extract power (bhp) from strings like '88.50bhp@6000rpm'."""
    match = re.search(r"([\d\.]+)bhp", str(text))
    return float(match.group(1)) if match else np.nan


if "max_torque" in df.columns:
    df["torque_nm"] = df["max_torque"].apply(extract_torque_value)
    df = df.drop(columns=["max_torque"])

if "max_power" in df.columns:
    df["power_bhp"] = df["max_power"].apply(extract_power_value)
    df = df.drop(columns=["max_power"])

# Step 4: Convert binary 'Yes'/'No' columns to 1 and 0
print("\n--- Step 3: Encoding Binary Features ---")
binary_columns = [
    col for col in df.columns if col.startswith("is_") and col != "is_claim"
]

for col in binary_columns:
    if df[col].dtype == "object":
        df[col] = df[col].map({"Yes": 1, "No": 0})

# Log-transform right-skewed feature
if "population_density" in df.columns:
    df["population_density"] = np.log1p(df["population_density"])

# Step 5: Handle categorical features using One-Hot Encoding (pd.get_dummies)
print("\n--- Step 4: One-Hot Encoding Categorical Columns ---")
X = df.drop(columns=["is_claim"])
y = df["is_claim"]

# ==========================================

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns found: {categorical_cols}")

# Apply get_dummies
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Fill any missing values with median for numeric columns
X_encoded = X_encoded.fillna(X_encoded.median())

# Step 6: Split Data into Train and Validation sets (80% Train, 20% Val)
print("\n--- Step 5: Splitting Data into Train & Validation ---")
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================

# Scale numeric features using StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns
)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

# Step 7: Save preprocessed data and scaler for future use
print("\n--- Step 6: Saving Processed Artifacts ---")

# Define and create output directory for artifacts
PROCESSED_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

saved_artifacts = {
    "X_train": X_train_scaled,
    "X_val": X_val_scaled,
    "y_train": y_train,
    "y_val": y_val,
    "scaler": scaler,
    "feature_names": X_encoded.columns.tolist(),
}

joblib.dump(saved_artifacts, PROCESSED_DIR / "split_data.pkl")
print(
    f"SUCCESS! Cleaned data saved to: {PROCESSED_DIR / 'split_data.pkl'}"
)

# ==========================================
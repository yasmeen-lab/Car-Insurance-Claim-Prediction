# Navigate to project directory
cd /home/dell/Documents/Final_project_Car_Insurance_Claim_Prediction

# Activate virtual environment
source .venv/bin/activate

# Upgrade package manager and install dependencies
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn joblib lightgbm xgboost catboost




# Step 1: Run Exploratory Data Analysis
python3 scripts/eda.py

# Step 2: Run Preprocessing & Feature Engineering
python3 scripts/data_preprocessing.py

# Step 3: Train and Benchmark Models
python3 scripts/train_models.py

# Step 4: Run Model Explainability & Diagnostics
python3 scripts/testing_models.py

# Step 5: Generate Final Test Set Predictions
python3 scripts/model_explanation.py
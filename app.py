# ==============================================================================
# Streamlit Dashboard: Car Insurance Claim Prediction
# Author: AI/ML Trainee
# Description: Real-time claim risk inference calculator & model performance analytics.
# ==============================================================================

import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Insurance Claim Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Load Model Artifacts (Cached for speed)
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "split_data.pkl"
BENCHMARK_CSV_PATH = BASE_DIR / "models" / "model_benchmark_results.csv"
FEATURE_IMP_PATH = BASE_DIR / "reports" / "feature_importances.csv"


@st.cache_resource
def load_artifacts():
    if not BEST_MODEL_PATH.exists() or not PROCESSED_DATA_PATH.exists():
        return None, None
    model = joblib.load(BEST_MODEL_PATH)
    data = joblib.load(PROCESSED_DATA_PATH)
    return model, data


best_model, data_artifacts = load_artifacts()

# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/car-insurance.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🎯 Single Claim Predictor", "📊 Model Performance Analytics", "📁 Batch Prediction (CSV)"],
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** Car Insurance Claim Risk Prediction  
**Champion Model:** Loaded from `best_model.pkl`  
**Metric:** ROC-AUC
""")

# ------------------------------------------------------------------------------
# PAGE 1: Single Claim Predictor
# ------------------------------------------------------------------------------
if page == "🎯 Single Claim Predictor":
    st.markdown('<div class="main-header">🚗 Policy Claim Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter policyholder and vehicle details to calculate real-time claim probability.</div>', unsafe_allow_html=True)

    if best_model is None or data_artifacts is None:
        st.error("Model artifacts not found! Please ensure `scripts/01_preprocess.py` and `scripts/02_train_models.py` have been executed.")
        st.stop()

    scaler = data_artifacts["scaler"]
    train_features = data_artifacts["feature_names"]

    with st.form("prediction_form"):
        st.subheader("1. Policyholder & Location Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            age_of_policyholder = st.slider("Policyholder Age (Normalized / Years)", 0.0, 1.0, 0.45, 0.01)
        with col2:
            population_density = st.number_input("Population Density", min_value=100, max_value=100000, value=20000, step=500)
        with col3:
            policy_tenure = st.slider("Policy Tenure", 0.0, 1.0, 0.5, 0.01)

        st.subheader("2. Vehicle Specifications")
        col4, col5, col6 = st.columns(3)
        with col4:
            age_of_car = st.slider("Car Age (Years / Normalized)", 0.0, 1.0, 0.1, 0.01)
            segment = st.selectbox("Vehicle Segment", ["A", "B1", "B2", "C1", "C2", "Utility"])
            fuel_type = st.selectbox("Fuel Type", ["CNG", "Petrol", "Diesel"])
        with col5:
            steering_type = st.selectbox("Steering Type", ["Manual", "Power", "Electric"])
            rear_brakes_type = st.selectbox("Rear Brakes Type", ["Drum", "Disc"])
            transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])
        with col6:
            ncap_rating = st.selectbox("NCAP Safety Rating", [0, 1, 2, 3, 4, 5])
            max_torque_val = st.number_input("Max Torque (Nm)", min_value=50.0, max_value=600.0, value=113.0)
            max_power_val = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=88.5)

        st.subheader("3. Safety Features & Flags")
        col7, col8, col9 = st.columns(3)
        with col7:
            is_esc = st.checkbox("Electronic Stability Control (ESC)")
            is_tpms = st.checkbox("Tire Pressure Monitoring (TPMS)")
            is_parking_sensors = st.checkbox("Rear Parking Sensors", value=True)
        with col8:
            is_parking_camera = st.checkbox("Rear Parking Camera")
            is_front_fog_lights = st.checkbox("Front Fog Lights", value=True)
            is_rear_window_defogger = st.checkbox("Rear Window Defogger")
        with col9:
            is_brake_assist = st.checkbox("Brake Assist", value=True)
            is_power_door_locks = st.checkbox("Power Door Locks", value=True)
            is_central_locking = st.checkbox("Central Locking", value=True)

        submit_button = st.form_submit_button("⚡ Predict Claim Risk", use_container_width=True)

    if submit_button:
        # Construct input raw dataframe
        input_dict = {
            "age_of_policyholder": age_of_policyholder,
            "population_density": np.log1p(population_density),
            "policy_tenure": policy_tenure,
            "age_of_car": age_of_car,
            "segment": segment,
            "fuel_type": fuel_type,
            "steering_type": steering_type,
            "rear_brakes_type": rear_brakes_type,
            "transmission_type": transmission_type,
            "ncap_rating": ncap_rating,
            "torque_nm": max_torque_val,
            "power_bhp": max_power_val,
            "is_esc": 1 if is_esc else 0,
            "is_tpms": 1 if is_tpms else 0,
            "is_parking_sensors": 1 if is_parking_sensors else 0,
            "is_parking_camera": 1 if is_parking_camera else 0,
            "is_front_fog_lights": 1 if is_front_fog_lights else 0,
            "is_rear_window_defogger": 1 if is_rear_window_defogger else 0,
            "is_brake_assist": 1 if is_brake_assist else 0,
            "is_power_door_locks": 1 if is_power_door_locks else 0,
            "is_central_locking": 1 if is_central_locking else 0,
        }

        input_df = pd.DataFrame([input_dict])

        # One-hot encode & align with training feature set
        categorical_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
        encoded_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        aligned_df = encoded_df.reindex(columns=train_features, fill_value=0)

        # Scale features
        scaled_input = scaler.transform(aligned_df)

        # Run inference
        prob = best_model.predict_proba(scaled_input)[0][1] if hasattr(best_model, "predict_proba") else 0.5
        pred = best_model.predict(scaled_input)[0]

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred == 1:
                st.error("⚠️ **HIGH CLAIM RISK DETECTED**")
            else:
                st.success("✅ **LOW CLAIM RISK**")

            st.metric(label="Predicted Claim Probability", value=f"{prob * 100:.2f}%")

        with res_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Index Gauge"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 30], 'color': "#D1FAE5"},
                        {'range': [30, 60], 'color': "#FEF3C7"},
                        {'range': [60, 100], 'color': "#FEE2E2"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50,
                    },
                },
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# PAGE 2: Model Performance Analytics
# ------------------------------------------------------------------------------
elif page == "📊 Model Performance Analytics":
    st.markdown('<div class="main-header">📊 Model Performance & Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics across candidate algorithms and global feature importances.</div>', unsafe_allow_html=True)

    if BENCHMARK_CSV_PATH.exists():
        benchmark_df = pd.read_csv(BENCHMARK_CSV_PATH)

        st.subheader("🏆 Model Leaderboard")
        st.dataframe(
            benchmark_df.style.highlight_max(subset=["ROC-AUC", "F1-Score", "Accuracy"], color="#D1FAE5"),
            use_container_width=True,
        )

        fig_bench = px.bar(
            benchmark_df,
            x="Model Name",
            y="ROC-AUC",
            color="ROC-AUC",
            title="ROC-AUC Comparison Across Models",
            color_continuous_scale="Blues",
            text="ROC-AUC",
        )
        fig_bench.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_bench.update_layout(yaxis_range=[0, 1.0])
        st.plotly_chart(fig_bench, use_container_width=True)

    else:
        st.warning("Benchmark CSV file not found at `models/model_benchmark_results.csv`.")

    if FEATURE_IMP_PATH.exists():
        st.subheader("⭐ Top 15 Feature Importances")
        feat_df = pd.read_csv(FEATURE_IMP_PATH).head(15)

        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Factors Driving Claim Probability",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig_feat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------------------------------------------
# PAGE 3: Batch Prediction (CSV)
# ------------------------------------------------------------------------------
elif page == "📁 Batch Prediction (CSV)":
    st.markdown('<div class="main-header">📁 Batch Inference Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an unlabelled CSV file to score multiple policies simultaneously.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded dataset preview ({len(batch_df)} rows):")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🚀 Process Batch Predictions"):
            st.info("Applying preprocessing and generating predictions...")

            policy_ids = batch_df["policy_id"] if "policy_id" in batch_df.columns else batch_df.index
            processed_df = batch_df.copy()

            if "policy_id" in processed_df.columns:
                processed_df = processed_df.drop(columns=["policy_id"])

            scaler = data_artifacts["scaler"]
            train_features = data_artifacts["feature_names"]

            # Extraction
            if "max_torque" in processed_df.columns:
                processed_df["torque_nm"] = processed_df["max_torque"].apply(
                    lambda x: float(re.search(r"([\d\.]+)Nm", str(x)).group(1)) if re.search(r"([\d\.]+)Nm", str(x)) else np.nan
                )
                processed_df = processed_df.drop(columns=["max_torque"])

            if "max_power" in processed_df.columns:
                processed_df["power_bhp"] = processed_df["max_power"].apply(
                    lambda x: float(re.search(r"([\d\.]+)bhp", str(x)).group(1)) if re.search(r"([\d\.]+)bhp", str(x)) else np.nan
                )
                processed_df = processed_df.drop(columns=["max_power"])

            binary_cols = [c for c in processed_df.columns if c.startswith("is_")]
            for col in binary_cols:
                if processed_df[col].dtype == "object":
                    processed_df[col] = processed_df[col].map({"Yes": 1, "No": 0})

            if "population_density" in processed_df.columns:
                processed_df["population_density"] = np.log1p(processed_df["population_density"])

            cat_cols = processed_df.select_dtypes(include=["object"]).columns.tolist()
            encoded_df = pd.get_dummies(processed_df, columns=cat_cols, drop_first=True)
            aligned_df = encoded_df.reindex(columns=train_features, fill_value=0).fillna(0)

            scaled_batch = scaler.transform(aligned_df)

            preds = best_model.predict(scaled_batch)
            probs = best_model.predict_proba(scaled_batch)[:, 1] if hasattr(best_model, "predict_proba") else preds

            out_df = pd.DataFrame({
                "policy_id": policy_ids,
                "is_claim_predicted": preds,
                "claim_probability": np.round(probs, 4),
            })

            st.success("Batch predictions completed!")
            st.dataframe(out_df.head(10), use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Predictions CSV",
                data=csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
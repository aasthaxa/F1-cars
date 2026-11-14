import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="F1 Analytics Dashboard",
    page_icon="🏎️",
    layout="wide"
)



# Load saved model bundle
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

bundle = load_model()

# Navigation
# Navigation
st.sidebar.title("🏎️ F1 Analytics")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Podium Prediction",
        "Model Comparison & Correlation",
        "Cluster Analysis",
        "About Project"
    ]
)
    st.image(
        "https://static01.nyt.com/images/2026/03/02/multimedia/02sp-preview-year-inyt-vhzq/02sp-preview-year-inyt-vhzq-articleLarge.jpg?quality=75&auto=avif&disable=upscale",
        width="stretch"
    )


# HOME PAGE
if page == "Home":
    st.title("🏎️ Formula 1 Analytics & Prediction Dashboard")
    st.write("Interactive Machine Learning & Statistical Performance Hub")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Supervised Classifiers", "6 (including Logistic Regression)")
    with col2:
        st.metric("Unsupervised Clusters", "4 Distinct Segments")
    with col3:
        st.metric("Evaluation Metrics", "F1-Score, Precision, Recall, Confusion Matrix")

    st.subheader("Project Overview")
    st.write("""
    This dashboard brings together statistical classification models and clustering algorithms to assess Formula 1 race trajectories.
    By assessing historic telemetry such as qualifying grid parameters, race timelines, and driver-constructor pairings, team leaders and strategists can predict potential podium finishes.
    """)
    st.info("Explore different pages on the sidebar to perform predictions, evaluate metrics, or inspect drivers' segments.")

# PREDICTION PAGE
elif page == "Podium Prediction":
    st.title("🎯 Live Podium Prediction")
    st.write("Provide starting parameters below to run real-time inference across our trained estimators.")

    grid = st.number_input("Starting Grid Position", min_value=1, max_value=40, value=5)
    race_id = st.number_input("Race ID", min_value=1, value=100)
    driver_id = st.number_input("Driver ID", min_value=1, value=1)
    constructor_id = st.number_input("Constructor ID", min_value=1, value=1)

    selected_model = st.selectbox(
        "Select Model for Prediction",
        list(bundle['models'].keys())
    )

    if st.button("Run Inference"):
        input_df = pd.DataFrame({
            'grid': [grid],
            'raceId': [race_id],
            'driverId': [driver_id],
            'constructorId': [constructor_id]
        })

        processed = bundle['preprocessor'].transform(input_df)
        model = bundle['models'][selected_model]
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        st.subheader("Predictive Assessment")
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success("🎉 Podium Finish Predicted!")
            else:
                st.warning("❌ No Podium Finish Predicted.")
        with col2:
            st.metric("Calculated Probability", f"{probability*100:.2f}%")
            st.progress(float(probability))

# MODEL COMPARISON & CORRELATION
elif page == "Model Comparison & Correlation":
    st.title("📊 Model Evaluation Hub")
    
    if os.path.exists("model_comparison.csv"):
        results_df = pd.read_csv("model_comparison.csv")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("F1-Score Breakdown (Imbalanced Class Metric)")
        st.bar_chart(results_df.set_index("Model")["F1 Score"])
    else:
        st.error("Comparison metrics CSV not found. Please train models first!")

# CLUSTER ANALYSIS
elif page == "Cluster Analysis":
    st.title("🔍 Unsupervised K-Means Segmentation")
    if os.path.exists("cluster_summary.csv"):
        cluster_df = pd.read_csv("cluster_summary.csv")
        st.dataframe(cluster_df, use_container_width=True)

        st.subheader("Podium Yield by Driver Segment")
        st.bar_chart(cluster_df.set_index("Cluster")["Podium_Rate"])
    else:
        st.error("Cluster summary CSV not found.")

# ABOUT PROJECT
elif page == "About Project":
    st.title("📝 Technical Architecture & Summary")
    st.markdown("""
    * **Feature Reduction:** TruncatedSVD components maintain categorical data density.
    * **Core Pipeline:** Imputation -> Standard Scaling -> SVD -> Classifier / Cluster Model.
    * **Key Finding:** Regression outputs show starting grid is the highest correlated predictor for a Top 3 Finish.
    """)

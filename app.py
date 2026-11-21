import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    bundle = joblib.load("smartguard_model_bundle.pkl")

    return (
        bundle["model"],
        bundle["preprocessor"],
        bundle["feature_names"],
        bundle["risk_thresholds"]
    )

try:
    model, preprocessor, feature_names, thresholds = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

# --------------------------------------------------
# 3. APPLICATION DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #F7F9FC;
}
h1, h2, h3 {
    color: #17365D;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛡️ SmartGuard")
st.sidebar.caption("Intelligent Credit Card Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Transaction Detection",
        "Risk Analysis",
        "Model Information"
    ]
)

# --------------------------------------------------
# 4. COMMON PREDICTION FUNCTION
# --------------------------------------------------

def analyze_transactions(data):

    # Keep the same feature order used during training
    missing = [
        col for col in feature_names
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features. "
            f"Examples: {missing[:10]}"
        )

    input_data = data[feature_names].copy()

    # Apply saved preprocessing
    processed_data = preprocessor.transform(input_data)

    # Predict fraud probability
    probabilities = model.predict_proba(processed_data)[:, 1]

    predictions = model.predict(processed_data)

    # Intelligent risk scoring
    scores = np.round(probabilities * 100, 2)

    risk_levels = []

    for score in scores:
        if score < thresholds["low"]:
            risk_levels.append("Low")
        elif score < thresholds["medium"]:
            risk_levels.append("Medium")
        else:
            risk_levels.append("High")

    results = data.copy()

    results["Fraud Probability (%)"] = scores
    results["Risk Score"] = scores
    results["Risk Level"] = risk_levels
    results["Prediction"] = np.where(
        predictions == 1, "Fraud", "Genuine"
    )

    return results, processed_data


# --------------------------------------------------
# 5. HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("🛡️ SmartGuard")
    st.subheader("Intelligent Credit Card Fraud Detection System")

    st.write(
        "SmartGuard uses machine learning to identify potentially "
        "fraudulent credit card transactions and estimate their risk."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Transactions", "590,540")
    col2.metric("Selected Model", "Tuned XGBoost")
    col3.metric("Input Features", "423")

    st.subheader("Project Features")

    st.markdown("""
    - **Fraud Detection:** Predicts whether a transaction is potentially fraudulent.
    - **SHAP Explanations:** Explains important factors behind model predictions.
    - **Intelligent Risk Scoring:** Converts fraud probability into a score from 0 to 100.
    - **Financial Loss Estimation:** Estimates the transaction amount associated with missed fraud.
    """)

    st.info(
        "SmartGuard provides decision support. Its predictions are "
        "not proof that a transaction is fraudulent."
    )

# --------------------------------------------------
# 6. TRANSACTION DETECTION PAGE
# --------------------------------------------------

elif page == "Transaction Detection":

    st.title("Transaction Detection")

    st.write(
        "Upload a CSV file containing the transaction features "
        "required by the trained model."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="transaction_upload"
    )

    if uploaded_file is not None:

        try:
            uploaded_data = pd.read_csv(uploaded_file)

            st.write("Uploaded data preview")
            st.dataframe(uploaded_data.head())

            if st.button("Analyze Transactions"):

                with st.spinner("Analyzing transactions..."):

                    results, processed_data = analyze_transactions(
                        uploaded_data
                    )

                    st.session_state["results"] = results
                    st.session_state["processed_data"] = processed_data
                    st.session_state["original_data"] = uploaded_data

                st.success("Transaction analysis completed!")

                fraud_count = (
                    results["Prediction"] == "Fraud"
                ).sum()

                genuine_count = (
                    results["Prediction"] == "Genuine"
                ).sum()

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Transactions", len(results))
                col2.metric("Potential Fraud", int(fraud_count))
                col3.metric("Genuine Predictions", int(genuine_count))

                st.subheader("Prediction Results")

                st.dataframe(results, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv,
                    file_name="smartguard_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Analysis error: {e}")

    st.caption(
        "The CSV must contain all 423 model input features. "
        "TransactionID and isFraud are not required as model inputs."
    )

# --------------------------------------------------
# 7. RISK ANALYSIS PAGE
# --------------------------------------------------

elif page == "Risk Analysis":

    st.title("Risk Analysis")

    if "results" not in st.session_state:

        st.info(
            "First upload and analyze transactions from the "
            "Transaction Detection page."
        )

    else:

        results = st.session_state["results"]
        processed_data = st.session_state["processed_data"]

        selected_index = st.selectbox(
            "Select transaction",
            range(len(results)),
            format_func=lambda i: f"Transaction {i + 1}"
        )

        selected = results.iloc[selected_index]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk Score",
            f"{selected['Risk Score']}/100"
        )

        col2.metric(
            "Risk Level",
            selected["Risk Level"]
        )

        col3.metric(
            "Prediction",
            selected["Prediction"]
        )

        st.subheader("SHAP-Based Fraud Explanation")

        if st.button("Generate SHAP Explanation"):

            try:
                with st.spinner("Generating explanation..."):

                    explainer = shap.TreeExplainer(model)

                    shap_values = explainer.shap_values(
                        processed_data[selected_index:selected_index + 1]
                    )

                    if isinstance(shap_values, list):
                        values = np.asarray(shap_values[-1])[0]
                    else:
                        values = np.asarray(shap_values)

                        if values.ndim == 3:
                            values = values[0, :, -1]
                        elif values.ndim == 2:
                            values = values[0]

                    transformed_names = (
                        preprocessor.get_feature_names_out()
                    )

                    explanation = pd.DataFrame({
                        "Feature": transformed_names,
                        "SHAP Impact": values
                    })

                    explanation["Direction"] = np.where(
                        explanation["SHAP Impact"] > 0,
                        "Pushes toward fraud",
                        "Pushes away from fraud"
                    )

                    explanation["Absolute Impact"] = (
                        explanation["SHAP Impact"].abs()
                    )

                    explanation = explanation.sort_values(
                        "Absolute Impact",
                        ascending=False
                    ).head(10)

                    st.write("Top 10 influential features")

                    st.dataframe(
                        explanation[
                            ["Feature", "SHAP Impact", "Direction"]
                        ],
                        use_container_width=True
                    )

                    fig, ax = plt.subplots(figsize=(9, 5))

                    plot_data = explanation.sort_values(
                        "SHAP Impact"
                    )

                    ax.barh(
                        plot_data["Feature"],
                        plot_data["SHAP Impact"]
                    )

                    ax.set_xlabel("SHAP Impact")
                    ax.set_title("Factors Influencing This Prediction")

                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.error(f"SHAP explanation error: {e}")

        st.caption(
            "Positive SHAP values push the model toward a fraud prediction; "
            "negative values push it away. These values are not percentages."
        )

# --------------------------------------------------
# 8. MODEL INFORMATION PAGE
# --------------------------------------------------

elif page == "Model Information":

    st.title("Model Information")

    st.subheader("Selected Machine Learning Model")

    st.write(
        "SmartGuard uses a tuned XGBoost classifier for credit card "
        "fraud detection."
    )

    st.subheader("Preliminary Tuned Model Performance")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Tuned XGBoost": [
            0.9508,
            0.2841,
            0.2979,
            0.2909,
            0.7512,
            0.2261
        ]
    })

    st.dataframe(metrics, hide_index=True)

    st.caption(
        "These are preliminary results from the 20,000-row development "
        "test sample, not a full-dataset benchmark."
    )

    st.subheader("Financial Loss Estimation")

    st.write(
        "The following estimates were obtained from the notebook's "
        "development test sample."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Missed Fraud Transactions",
        "476"
    )

    col2.metric(
        "Estimated Missed-Fraud Exposure",
        "74,927.68"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Genuine Transactions Flagged",
        "509"
    )

    col4.metric(
        "Amount of False Alerts",
        "87,616.98"
    )

    st.warning(
        "These amounts represent transaction-value estimates, not "
        "confirmed financial losses. They are based on the development "
        "test sample and do not represent actual chargebacks or "
        "investigation costs."
    )

    st.subheader("Risk Score Categories")

    st.write("Low Risk: 0–34.99")
    st.write("Medium Risk: 35–69.99")
    st.write("High Risk: 70–100")

    st.caption(
        "Risk thresholds are provisional and should be calibrated "
        "before real-world use."
    )

# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "SmartGuard | CSE303 Capstone Project"
)

st.sidebar.caption(
    "Machine Learning-Based Fraud Detection"
)

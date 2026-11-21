
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
    st.error(f"Model loading failed: {e}")
    st.stop()

# --------------------------------------------------
# 3. PREPARE DEFAULT VALUES FOR MANUAL ENTRY
# --------------------------------------------------

def create_default_transaction():

    defaults = {}

    for name, transformer, columns in preprocessor.transformers_:

        if name == "remainder" or transformer == "drop":
            continue

        if name == "num":
            imputer = transformer.named_steps["imputer"]

        elif name == "cat":
            imputer = transformer.named_steps["imputer"]

        else:
            continue

        for column, value in zip(columns, imputer.statistics_):
            defaults[column] = value

    row = {
        column: defaults.get(column, np.nan)
        for column in feature_names
    }

    return pd.DataFrame([row], columns=feature_names)


# --------------------------------------------------
# 4. COMMON PREDICTION FUNCTION
# --------------------------------------------------

def analyze_transactions(data):

    missing = [
        col for col in feature_names
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} required features. "
            f"Examples: {missing[:10]}"
        )

    input_data = data[feature_names].copy()

    processed_data = preprocessor.transform(input_data)

    probabilities = model.predict_proba(processed_data)[:, 1]

    predictions = model.predict(processed_data)

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
        predictions == 1,
        "Potential Fraud",
        "Likely Genuine"
    )

    return results, processed_data


# --------------------------------------------------
# 5. APPLICATION DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
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
# 6. HOME PAGE
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
    col3.metric("Model Input Features", "423")

    st.subheader("SmartGuard Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔍 Fraud Detection**

        Predicts whether a transaction may be fraudulent.

        **📊 Intelligent Risk Scoring**

        Converts predicted fraud probability into a score from 0 to 100.
        """)

    with col2:
        st.markdown("""
        **🧠 SHAP Explanation**

        Explains the important factors influencing a prediction.

        **💰 Financial Loss Estimation**

        Estimates the transaction value associated with missed fraud.
        """)

    st.info(
        "SmartGuard is a machine-learning decision-support demonstration. "
        "Its predictions are not proof of fraud."
    )

# --------------------------------------------------
# 7. TRANSACTION DETECTION PAGE
# --------------------------------------------------

elif page == "Transaction Detection":

    st.title("Transaction Detection")

    st.write(
        "Enter a transaction manually or upload a CSV file "
        "to receive a fraud prediction."
    )

    manual_tab, upload_tab = st.tabs([
        "✍️ Manual Transaction Entry",
        "📁 Upload CSV"
    ])

    # ----------------------------------------------
    # A. MANUAL TRANSACTION ENTRY
    # ----------------------------------------------

    with manual_tab:

        st.subheader("Enter Transaction Details")

        st.caption(
            "The model requires 423 input features. This simplified "
            "form fills the other features using training-data "
            "imputation defaults. Its result is an approximate "
            "demonstration, not a full transaction assessment."
        )

        with st.form("manual_transaction_form"):

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.01,
                value=100.0,
                step=10.0
            )

            col1, col2 = st.columns(2)

            with col1:
                product = st.selectbox(
                    "Product Type",
                    ["W", "C", "R", "H", "S"]
                )

            with col2:
                card_type = st.selectbox(
                    "Card Network",
                    [
                        "visa",
                        "mastercard",
                        "american express",
                        "discover"
                    ]
                )

            card_category = st.selectbox(
                "Card Category",
                ["debit", "credit", "charge"]
            )

            submitted = st.form_submit_button(
                "Predict Fraud Risk",
                use_container_width=True
            )

        if submitted:

            try:
                manual_data = create_default_transaction()

                if "TransactionAmt" in manual_data.columns:
                    manual_data.loc[0, "TransactionAmt"] = amount

                if "ProductCD" in manual_data.columns:
                    manual_data.loc[0, "ProductCD"] = product

                if "card4" in manual_data.columns:
                    manual_data.loc[0, "card4"] = card_type

                if "card6" in manual_data.columns:
                    manual_data.loc[0, "card6"] = card_category

                results, processed_data = analyze_transactions(
                    manual_data
                )

                st.session_state["results"] = results
                st.session_state["processed_data"] = processed_data
                st.session_state["original_data"] = manual_data

                result = results.iloc[0]

                st.divider()
                st.subheader("Transaction Prediction")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Fraud Probability",
                    f"{result['Fraud Probability (%)']}%"
                )

                col2.metric(
                    "Risk Score",
                    f"{result['Risk Score']}/100"
                )

                col3.metric(
                    "Risk Level",
                    result["Risk Level"]
                )

                if result["Prediction"] == "Potential Fraud":
                    st.error("The model predicts potential fraud.")
                else:
                    st.success("The model predicts a likely genuine transaction.")

                st.warning(
                    "This prediction uses default values for the other "
                    "model features. It is for demonstration only."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ----------------------------------------------
    # B. CSV UPLOAD
    # ----------------------------------------------

    with upload_tab:

        st.subheader("Upload Transaction CSV")

        sample_path = Path("transaction.csv")

        if sample_path.exists():

            sample_data = sample_path.read_bytes()

            st.download_button(
                "⬇️ Download Sample transaction.csv",
                data=sample_data,
                file_name="transaction.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "The original sample transaction.csv has not yet "
                "been added to the GitHub repository."
            )

            template = create_default_transaction()

            template_csv = template.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Sample CSV Template",
                data=template_csv,
                file_name="transaction_template.csv",
                mime="text/csv",
                use_container_width=True
            )

        uploaded_file = st.file_uploader(
            "Choose a transaction CSV file",
            type=["csv"],
            key="transaction_upload"
        )

        if uploaded_file is not None:

            try:
                uploaded_data = pd.read_csv(uploaded_file)

                st.write("Uploaded transaction preview")
                st.dataframe(uploaded_data.head())

                if st.button("Analyze Uploaded Transactions"):

                    results, processed_data = analyze_transactions(
                        uploaded_data
                    )

                    st.session_state["results"] = results
                    st.session_state["processed_data"] = processed_data
                    st.session_state["original_data"] = uploaded_data

                    st.success("Transaction analysis completed!")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Transactions",
                        len(results)
                    )

                    col2.metric(
                        "Potential Fraud",
                        int((results["Prediction"] == "Potential Fraud").sum())
                    )

                    col3.metric(
                        "Likely Genuine",
                        int((results["Prediction"] == "Likely Genuine").sum())
                    )

                    st.subheader("Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    csv = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Prediction Results",
                        data=csv,
                        file_name="smartguard_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"CSV analysis error: {e}")

        st.caption(
            "For full-feature predictions, the uploaded CSV must contain "
            "all 423 model input features. Do not include the isFraud "
            "target column as an input."
        )

# --------------------------------------------------
# 8. RISK ANALYSIS PAGE
# --------------------------------------------------

elif page == "Risk Analysis":

    st.title("Risk Analysis")

    if "results" not in st.session_state:

        st.info(
            "First enter a transaction or upload a CSV "
            "from the Transaction Detection page."
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
                        processed_data[
                            selected_index:selected_index + 1
                        ]
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
            "Positive SHAP values push the model toward fraud. "
            "Negative values push it away. SHAP values are not percentages."
        )

# --------------------------------------------------
# 9. MODEL INFORMATION PAGE
# --------------------------------------------------

elif page == "Model Information":

    st.title("Model Information")

    st.subheader("Selected Machine Learning Model")

    st.write(
        "SmartGuard uses a tuned XGBoost classifier "
        "for credit card fraud detection."
    )

    st.subheader("Preliminary Model Performance")

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
        "Preliminary results from the 20,000-row development test sample, "
        "not a full-dataset benchmark."
    )

    st.subheader("Financial Loss Estimation")

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
        "These values are estimates from the development test sample. "
        "They represent transaction amounts, not confirmed financial losses."
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
# 10. SIDEBAR FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption("SmartGuard | CSE303 Capstone Project")
st.sidebar.caption("Machine Learning-Based Fraud Detection")

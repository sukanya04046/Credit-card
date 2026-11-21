
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LIGHT BLUE THEME
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #EAF3F8;
}

[data-testid="stHeader"] {
    background-color: #EAF3F8;
}

[data-testid="stSidebar"] {
    background-color: #D8EAF4;
}

[data-testid="stSidebar"] * {
    color: #17365D;
}

h1, h2, h3 {
    color: #17365D !important;
}

p, label, span {
    color: #263B50;
}

.stButton > button {
    background-color: #0F9D9A;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #087F7D;
    color: white;
    border: none;
}

.stDownloadButton > button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #D2E3ED;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    background-color: white;
}

div[data-testid="stForm"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #D2E3ED;
}

hr {
    border-color: #C5DCE9;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL BUNDLE
# --------------------------------------------------

@st.cache_resource
def load_model():
    bundle = joblib.load("smartguard_model_bundle.pkl")

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]
    thresholds = bundle.get(
        "risk_thresholds",
        {"low": 35, "medium": 70}
    )

    return model, preprocessor, feature_names, thresholds


try:
    model, preprocessor, feature_names, thresholds = load_model()
except Exception as e:
    st.error("Unable to load the SmartGuard model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# CREATE DEFAULT TRANSACTION
# --------------------------------------------------

def create_default_transaction():

    defaults = {
        feature: np.nan for feature in feature_names
    }

    for name, transformer, columns in preprocessor.transformers_:

        if name == "remainder":
            continue

        if isinstance(transformer, str):
            continue

        if not hasattr(transformer, "named_steps"):
            continue

        imputer = transformer.named_steps.get("imputer")

        if imputer is None:
            continue

        for column, value in zip(columns, imputer.statistics_):
            defaults[column] = value

    return pd.DataFrame([defaults], columns=feature_names)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def analyze_transactions(data):

    missing_columns = [
        column for column in feature_names
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Your CSV is missing {len(missing_columns)} required columns. "
            f"First missing columns: {missing_columns[:10]}"
        )

    data = data[feature_names].copy()

    processed_data = preprocessor.transform(data)

    probabilities = model.predict_proba(processed_data)[:, 1]

    results = data.copy()

    results["Fraud Probability (%)"] = np.round(
        probabilities * 100, 2
    )

    results["Risk Score"] = np.round(
        probabilities * 100, 2
    )

    results["Risk Level"] = [
        "Low" if score < thresholds["low"]
        else "Medium" if score < thresholds["medium"]
        else "High"
        for score in probabilities * 100
    ]

    results["Prediction"] = [
        "Potential Fraud" if probability >= 0.5
        else "Likely Genuine"
        for probability in probabilities
    ]

    return results, processed_data, probabilities


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

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

st.sidebar.divider()
st.sidebar.info(
    "SmartGuard uses a trained XGBoost model "
    "to estimate transaction fraud risk."
)


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("🛡️ SmartGuard")
    st.subheader("Intelligent Credit Card Fraud Detection")

    st.markdown("""
    SmartGuard analyses transaction patterns using machine learning
    to identify potentially fraudulent credit card transactions.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Transactions", "590,540")
    col2.metric("Selected Model", "XGBoost")
    col3.metric("Input Features", "423")

    st.divider()

    st.subheader("How SmartGuard Works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 1. Input")
        st.write("Enter transaction details or upload a CSV file.")

    with c2:
        st.markdown("### 2. Analysis")
        st.write("The trained model analyses transaction patterns.")

    with c3:
        st.markdown("### 3. Result")
        st.write("View the estimated fraud probability and risk level.")

    st.divider()

    st.subheader("Risk Classification")

    r1, r2, r3 = st.columns(3)

    r1.success("LOW RISK\n\nScore below 35")
    r2.warning("MEDIUM RISK\n\nScore from 35 to below 70")
    r3.error("HIGH RISK\n\nScore of 70 or above")

    st.caption(
        "Risk thresholds are configurable demonstration thresholds, "
        "not proof that a transaction is fraudulent."
    )


# ==================================================
# TRANSACTION DETECTION PAGE
# ==================================================

elif page == "Transaction Detection":

    st.title("Transaction Detection")
    st.write("Predict the estimated fraud risk of a transaction.")

    tab1, tab2 = st.tabs([
        "Manual Transaction Entry",
        "CSV Upload"
    ])

    # ----------------------------------------------
    # MANUAL ENTRY
    # ----------------------------------------------

    with tab1:

        st.subheader("Enter Transaction Details")

        st.info(
            "For this demonstration, the four entered fields are "
            "combined with default values for the remaining model "
            "features. This is an approximate prediction, not a "
            "complete real-world transaction assessment."
        )

        with st.form("manual_transaction_form"):

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            product = st.selectbox(
                "Product Code",
                ["W", "C", "R", "H", "S"]
            )

            card4 = st.selectbox(
                "Card Network",
                ["visa", "mastercard", "discover", "american express"]
            )

            card6 = st.selectbox(
                "Card Type",
                ["credit", "debit", "charge"]
            )

            submitted = st.form_submit_button(
                "Predict Fraud Risk",
                use_container_width=True
            )

        if submitted:

            manual_data = create_default_transaction()

            if "TransactionAmt" in manual_data.columns:
                manual_data.loc[0, "TransactionAmt"] = amount

            if "ProductCD" in manual_data.columns:
                manual_data.loc[0, "ProductCD"] = product

            if "card4" in manual_data.columns:
                manual_data.loc[0, "card4"] = card4

            if "card6" in manual_data.columns:
                manual_data.loc[0, "card6"] = card6

            try:
                results, processed, probabilities = analyze_transactions(
                    manual_data
                )

                st.session_state.results = results
                st.session_state.processed_data = processed
                st.session_state.probabilities = probabilities

                score = float(probabilities[0] * 100)

                st.divider()
                st.subheader("Prediction Result")

                a, b = st.columns(2)

                a.metric("Fraud Probability", f"{score:.2f}%")

                if score < thresholds["low"]:
                    b.success("LOW RISK")
                elif score < thresholds["medium"]:
                    b.warning("MEDIUM RISK")
                else:
                    b.error("HIGH RISK")

                st.write(
                    "Model prediction:",
                    results.iloc[0]["Prediction"]
                )

            except Exception as e:
                st.error("Prediction could not be completed.")
                st.exception(e)

    # ----------------------------------------------
    # CSV UPLOAD
    # ----------------------------------------------

    with tab2:

        st.subheader("Upload Transaction CSV")

        st.write(
            "Upload a CSV containing all 423 model input features. "
            "Do not include the target column isFraud."
        )

        # Download sample CSV
        if os.path.exists("transaction.csv"):

            with open("transaction.csv", "rb") as file:
                st.download_button(
                    label="Download Sample Transaction CSV",
                    data=file,
                    file_name="transaction.csv",
                    mime="text/csv"
                )

        else:

            sample = create_default_transaction()

            st.download_button(
                label="Download Sample CSV Template",
                data=sample.to_csv(index=False).encode("utf-8"),
                file_name="transaction_template.csv",
                mime="text/csv"
            )

        uploaded_file = st.file_uploader(
            "Choose a transaction CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:
                uploaded_data = pd.read_csv(uploaded_file)

                st.write("Uploaded data preview")
                st.dataframe(
                    uploaded_data.head(),
                    use_container_width=True
                )

                if st.button(
                    "Analyze Uploaded Transactions",
                    use_container_width=True
                ):

                    results, processed, probabilities = analyze_transactions(
                        uploaded_data
                    )

                    st.session_state.results = results
                    st.session_state.processed_data = processed
                    st.session_state.probabilities = probabilities

                    st.success("Transaction analysis completed!")

                    st.subheader("Prediction Results")

                    st.dataframe(
                        results[
                            [
                                "Fraud Probability (%)",
                                "Risk Score",
                                "Risk Level",
                                "Prediction"
                            ]
                        ],
                        use_container_width=True
                    )

                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Prediction Results",
                        data=csv_data,
                        file_name="smartguard_predictions.csv",
                        mime="text/csv"
                    )

                    st.metric(
                        "Transactions Analyzed",
                        len(results)
                    )

                    high_risk = (
                        results["Risk Level"] == "High"
                    ).sum()

                    st.metric(
                        "High-Risk Transactions",
                        int(high_risk)
                    )

            except Exception as e:
                st.error("Unable to process this CSV file.")
                st.exception(e)


# ==================================================
# RISK ANALYSIS PAGE
# ==================================================

elif page == "Risk Analysis":

    st.title("Risk Analysis")
    st.write(
        "Explore transaction risk scores and model explanations."
    )

    if st.session_state.results is None:

        st.info(
            "First analyze a transaction using the Transaction Detection page."
        )

    else:

        results = st.session_state.results
        processed = st.session_state.processed_data
        probabilities = st.session_state.probabilities

        st.subheader("Transaction Risk Summary")

        st.dataframe(
            results[
                [
                    "Fraud Probability (%)",
                    "Risk Score",
                    "Risk Level",
                    "Prediction"
                ]
            ],
            use_container_width=True
        )

        selected_index = st.selectbox(
            "Select transaction for explanation",
            range(len(results)),
            format_func=lambda x: f"Transaction {x + 1}"
        )

        st.divider()
        st.subheader("SHAP-Based Explanation")

        st.write(
            "SHAP helps explain which model features contributed "
            "to the selected prediction."
        )

        try:

            explainer = shap.TreeExplainer(model)

            selected_row = processed[selected_index]

            if hasattr(selected_row, "toarray"):
                selected_row = selected_row.toarray()

            shap_values = explainer.shap_values(selected_row)

            if isinstance(shap_values, list):
                values = np.asarray(shap_values[-1]).reshape(-1)
            else:
                values = np.asarray(shap_values)

                if values.ndim == 3:
                    values = values[0, :, -1]
                elif values.ndim == 2:
                    values = values[0]

                values = values.reshape(-1)

            try:
                transformed_names = preprocessor.get_feature_names_out()
            except Exception:
                transformed_names = [
                    f"Feature {i + 1}"
                    for i in range(len(values))
                ]

            explanation = pd.DataFrame({
                "Feature": transformed_names,
                "SHAP Impact": values
            })

            explanation["Absolute Impact"] = (
                explanation["SHAP Impact"].abs()
            )

            explanation = explanation.sort_values(
                "Absolute Impact",
                ascending=False
            ).head(10)

            st.write("Top 10 contributing features")

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.barh(
                explanation["Feature"][::-1],
                explanation["SHAP Impact"][::-1]
            )

            ax.set_xlabel("SHAP Value")
            ax.set_title("Top Feature Contributions")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "SHAP values explain model output contributions. "
                "They are not percentages or proof of fraud."
            )

        except Exception as e:
            st.warning(
                "SHAP explanation could not be generated for this prediction."
            )
            st.exception(e)


# ==================================================
# MODEL INFORMATION PAGE
# ==================================================

elif page == "Model Information":

    st.title("Model Information")

    st.subheader("Selected Machine Learning Model")

    st.success("Tuned XGBoost Classifier")

    st.write(
        "SmartGuard uses a trained XGBoost model with a saved "
        "preprocessing pipeline."
    )

    st.divider()

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

    st.dataframe(metrics, use_container_width=True)

    st.caption(
        "These are preliminary development-sample results, not "
        "a full-data benchmark. Performance depends on the "
        "validation sample and decision threshold."
    )

    st.divider()

    st.subheader("Financial Loss Estimation")

    col1, col2 = st.columns(2)

    col1.metric(
        "Missed Fraud Transactions",
        "476"
    )

    col2.metric(
        "Estimated Missed-Fraud Exposure",
        "$74,927.68"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Genuine Transactions Flagged",
        "509"
    )

    col4.metric(
        "Value of Flagged Genuine Transactions",
        "$87,616.98"
    )

    st.warning(
        "These amounts represent transaction values in the evaluation "
        "sample, not confirmed financial losses or recovered amounts."
    )

    st.divider()

    st.subheader("Project Features")

    st.markdown("""
    - Machine-learning-based fraud prediction
    - Intelligent risk scoring from 0 to 100
    - Low, Medium and High risk classification
    - CSV batch transaction analysis
    - SHAP-based model explanations
    - Financial exposure estimation
    - Manual transaction prediction
    """)

    st.divider()

    st.subheader("Important Limitations")

    st.markdown("""
    - A high risk score does not prove fraud.
    - Manual predictions use default values for fields not entered.
    - The model's performance figures are preliminary.
    - Financial exposure is an estimate based on missed-fraud transaction values.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<div style='text-align:center; color:#526B80;'>"
    "SmartGuard | Intelligent Credit Card Fraud Detection System"
    "</div>",
    unsafe_allow_html=True
)

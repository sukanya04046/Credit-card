
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SmartGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# LIGHT BLUE THEME
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #EAF3F8;
    color: #263B50;
}

[data-testid="stHeader"] {
    background-color: #EAF3F8;
}

[data-testid="stSidebar"] {
    background-color: #D8EAF4;
}

h1, h2, h3, h4, h5, h6 {
    color: #17365D !important;
}

.stApp p,
.stApp li,
.stApp label,
.stMarkdown p,
.stMarkdown li {
    color: #263B50;
}

/* Sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #17365D !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #C5DCE9;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #17365D !important;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #17365D !important;
    border-color: #A9C7D9 !important;
}

/* Dropdowns */
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #17365D !important;
}

[data-baseweb="select"] * {
    color: #17365D !important;
}

/* Buttons */
.stButton > button {
    background-color: #0F9D9A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #087F7D !important;
    color: #FFFFFF !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: #17365D !important;
    color: #FFFFFF !important;
}

/* Forms */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #C5DCE9;
    border-radius: 12px;
    padding: 20px;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #17365D !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #C5DCE9;
}

/* Radio buttons */
.stRadio label {
    color: #17365D !important;
}

/* Divider */
hr {
    border-color: #C5DCE9 !important;
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# REUSABLE COLORED INFORMATION CARDS
# ==================================================

def info_card(message):
    st.markdown(f"""
    <div style="
        background-color:#DCEFFA;
        padding:16px;
        border-radius:10px;
        border-left:5px solid #2878A5;
        margin:8px 0 16px 0;">
        <p style="color:#17365D !important;
        margin:0; line-height:1.6;">
        {message}
        </p>
    </div>
    """, unsafe_allow_html=True)


def success_card(title, message=""):
    st.markdown(f"""
    <div style="
        background-color:#DDF3E4;
        padding:16px;
        border-radius:10px;
        border-left:5px solid #16804A;
        margin:8px 0 16px 0;">
        <h3 style="color:#14532D !important; margin:0;">
        {title}
        </h3>
        <p style="color:#14532D !important; margin:5px 0 0 0;">
        {message}
        </p>
    </div>
    """, unsafe_allow_html=True)


def warning_card(message):
    st.markdown(f"""
    <div style="
        background-color:#FFF1CC;
        padding:16px;
        border-radius:10px;
        border-left:5px solid #C18A16;
        margin:8px 0 16px 0;">
        <p style="color:#754C00 !important;
        margin:0; line-height:1.6;">
        {message}
        </p>
    </div>
    """, unsafe_allow_html=True)


def risk_card(level, description, background, text_color, border_color):
    st.markdown(f"""
    <div style="
        background-color:{background};
        padding:18px;
        border-radius:12px;
        border:1px solid {border_color};
        border-left:5px solid {border_color};
        min-height:105px;">
        <h3 style="color:{text_color} !important; margin:0;">
        {level}
        </h3>
        <p style="color:{text_color} !important; margin-top:8px;">
        {description}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

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


# ==================================================
# CREATE DEFAULT TRANSACTION
# ==================================================

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

    return pd.DataFrame(
        [defaults],
        columns=feature_names
    )


# ==================================================
# TRANSACTION PREDICTION
# ==================================================

def analyze_transactions(data):

    missing_columns = [
        column for column in feature_names
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV is missing {len(missing_columns)} required columns. "
            f"Missing examples: {missing_columns[:10]}"
        )

    data = data[feature_names].copy()

    processed_data = preprocessor.transform(data)

    probabilities = model.predict_proba(
        processed_data
    )[:, 1]

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


# ==================================================
# SESSION STATE
# ==================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🛡️ SmartGuard")

st.sidebar.caption(
    "Intelligent Credit Card Fraud Detection"
)

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

st.sidebar.markdown("""
<div style="
    background-color:#EAF3F8;
    padding:14px;
    border-radius:10px;
    border:1px solid #A9C7D9;">
    <p style="color:#17365D !important; margin:0;
    line-height:1.6;">
    SmartGuard uses machine learning to estimate the risk
    of potentially fraudulent transactions.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("🛡️ SmartGuard")

    st.subheader(
        "Intelligent Credit Card Fraud Detection System"
    )

    st.write(
        "SmartGuard analyses transaction patterns using "
        "machine learning to identify potentially fraudulent "
        "credit card transactions."
    )

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
        st.write("The trained machine learning model analyses transaction patterns.")

    with c3:
        st.markdown("### 3. Result")
        st.write("View the estimated fraud probability and risk level.")

    st.divider()

    st.subheader("Risk Classification")

    r1, r2, r3 = st.columns(3)

    with r1:
        risk_card(
            "LOW RISK",
            "Score below 35",
            "#DDF3E4",
            "#14532D",
            "#70B98A"
        )

    with r2:
        risk_card(
            "MEDIUM RISK",
            "Score from 35 to below 70",
            "#FFF1CC",
            "#754C00",
            "#D5A83E"
        )

    with r3:
        risk_card(
            "HIGH RISK",
            "Score of 70 or above",
            "#FCE1E1",
            "#991B1B",
            "#D77D7D"
        )

    info_card(
        "Risk scores are estimates generated by the model. "
        "A high risk score does not prove that a transaction is fraudulent."
    )


# ==================================================
# TRANSACTION DETECTION
# ==================================================

elif page == "Transaction Detection":

    st.title("Transaction Detection")

    st.write(
        "Predict the estimated fraud risk of a transaction."
    )

    tab1, tab2 = st.tabs([
        "Manual Transaction Entry",
        "CSV Upload"
    ])

    # ----------------------------------------------
    # MANUAL ENTRY
    # ----------------------------------------------

    with tab1:

        st.subheader("Enter Transaction Details")

        info_card(
            "The four entered fields are combined with default "
            "values for the remaining model features. This is an "
            "approximate demonstration prediction, not a complete "
            "real-world transaction assessment."
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
                [
                    "visa",
                    "mastercard",
                    "discover",
                    "american express"
                ]
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

        info_card(
            "Upload a CSV containing all 423 model input features. "
            "Do not include the target column isFraud."
        )

        if os.path.exists("transaction.csv"):

            with open("transaction.csv", "rb") as file:

                st.download_button(
                    "Download Sample Transaction CSV",
                    data=file,
                    file_name="transaction.csv",
                    mime="text/csv"
                )

        else:

            sample = create_default_transaction()

            st.download_button(
                "Download Sample CSV Template",
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

                st.write("Uploaded Data Preview")

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
# RISK ANALYSIS
# ==================================================

elif page == "Risk Analysis":

    st.title("Risk Analysis")

    st.write(
        "Explore transaction risk scores and model explanations."
    )

    if st.session_state.results is None:

        info_card(
            "First analyze a transaction using the "
            "Transaction Detection page."
        )

    else:

        results = st.session_state.results
        processed = st.session_state.processed_data

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

        info_card(
            "SHAP explains which model features contributed "
            "to the selected prediction. It helps administrators "
            "understand the model's decision."
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

            info_card(
                "SHAP values explain model output contributions. "
                "They are not percentages or proof of fraud."
            )

        except Exception as e:

            st.error(
                "SHAP explanation could not be generated."
            )

            st.exception(e)


# ==================================================
# MODEL INFORMATION
# ==================================================

elif page == "Model Information":

    st.title("Model Information")

    st.subheader("Selected Machine Learning Model")

    success_card(
        "Tuned XGBoost Classifier",
        "Selected machine learning model used for SmartGuard transaction prediction."
    )

    info_card(
        "SmartGuard uses a trained XGBoost model with a saved "
        "preprocessing pipeline to estimate transaction fraud risk."
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

    st.dataframe(
        metrics,
        use_container_width=True
    )

    info_card(
        "These are preliminary development-sample results, "
        "not a full-data benchmark. Performance depends on "
        "the validation sample and decision threshold."
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

    warning_card(
        "These amounts represent transaction values in the "
        "evaluation sample, not confirmed financial losses "
        "or recovered amounts."
    )

    st.divider()

    st.subheader("Project Features")

    info_card(
        "SmartGuard includes machine-learning-based fraud prediction, "
        "intelligent risk scoring from 0 to 100, Low/Medium/High risk "
        "classification, CSV batch transaction analysis, SHAP-based "
        "model explanations, financial exposure estimation and "
        "manual transaction prediction."
    )

    st.divider()

    st.subheader("Important Limitations")

    warning_card(
        "A high risk score does not prove fraud. Manual predictions "
        "use default values for fields not entered. The model's "
        "performance figures are preliminary. Financial exposure "
        "is an estimate based on missed-fraud transaction values."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<div style="
    background-color:#D8EAF4;
    padding:16px;
    border-radius:10px;
    text-align:center;
    border:1px solid #A9C7D9;">
    <p style="color:#17365D !important; margin:0;">
    <strong>SmartGuard</strong><br>
    Intelligent Credit Card Fraud Detection System
    </p>
</div>
""", unsafe_allow_html=True)

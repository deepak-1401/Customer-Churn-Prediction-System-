"""
Customer Churn Prediction System - Streamlit App
=================================================
Run with:   streamlit run app.py
Requires:   streamlit, pandas, numpy, scikit-learn, plotly, joblib
Files expected in the same folder:
    model.pkl   -> trained Logistic Regression (best F1 = 63.4%)
    scaler.pkl  -> StandardScaler fitted on [tenure, MonthlyCharges, TotalCharges]
Optional:
    columns.pkl -> list of the exact feature columns used in training
                   (only needed if your model has no `feature_names_in_`)
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# 1. PAGE CONFIG + CUSTOM CSS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light CSS polish: card look for metrics, tighter headings, hidden footer.
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetric"] {
            background-color: rgba(120,120,120,0.08);
            border: 1px solid rgba(120,120,120,0.2);
            padding: 14px 18px;
            border-radius: 12px;
        }
        .driver-card {
            border-left: 5px solid #e74c3c;
            background-color: rgba(231,76,60,0.07);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .driver-card.safe {
            border-left-color: #2ecc71;
            background-color: rgba(46,204,113,0.07);
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 2. CONFIGURATION  (edit these if your training column names differ)
# ----------------------------------------------------------------------------
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]  # order used when fitting the scaler
RISK_THRESHOLD = 0.50  # probability above which a customer is flagged "High Risk"


# ----------------------------------------------------------------------------
# 3. LOAD ARTIFACTS (cached so they load only once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load model, scaler and (optionally) the training column list."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Work out the exact column order the model was trained on.
    if hasattr(model, "feature_names_in_"):          # set when trained on a DataFrame
        columns = list(model.feature_names_in_)
    elif os.path.exists("columns.pkl"):              # fallback: a saved list of columns
        columns = list(joblib.load("columns.pkl"))
    else:
        columns = None
    return model, scaler, columns


try:
    model, scaler, TRAIN_COLUMNS = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Could not find a required file: **{e.filename}**. "
             "Place `model.pkl` and `scaler.pkl` in the same folder as `app.py`.")
    st.stop()

# ----------------------------------------------------------------------------
# 4. SIDEBAR - CUSTOMER INPUTS (grouped into expanders)
# ----------------------------------------------------------------------------
st.sidebar.title("🧾 Customer Profile")
st.sidebar.caption("Enter the customer's details, then click **Predict**.")

with st.sidebar.form("customer_form"):

    # ---- Account information -------------------------------------------------
    with st.expander("📅 Account & Billing", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly Charges (₹)", 0.0, 200.0, 70.0, step=0.5)
        total = st.number_input(
            "Total Charges (₹)", 0.0, 10000.0,
            float(round(tenure * monthly, 2)), step=10.0,
            help="Defaults to tenure x monthly charges; overwrite if you know the real figure.",
        )

    # ---- Services ------------------------------------------------------------
    with st.expander("🌐 Services", expanded=False):
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    # ---- Demographics --------------------------------------------------------
    with st.expander("👤 Demographics", expanded=False):
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)


# ----------------------------------------------------------------------------
# 5. PREPROCESSING - make one row look exactly like the training data
# ----------------------------------------------------------------------------
def build_feature_row() -> pd.DataFrame:
    """Raw inputs -> one-hot encode -> align with training columns -> scale numerics."""
    raw = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,   # numeric 0/1 in the standard dataset
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    # One-hot encode all text columns -> e.g. "Contract_One year", "Partner_Yes"
    encoded = pd.get_dummies(raw)

    # Align with training columns: any dummy missing here (or dropped via
    # drop_first during training) is filled with 0, extras are discarded.
    if TRAIN_COLUMNS is not None:
        encoded = encoded.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    encoded = encoded.astype(float)

    # Scale ONLY the numeric columns with the scaler fitted on the training set.
    # (transform, never fit_transform -> no data leakage)
    encoded[NUMERIC_COLS] = scaler.transform(encoded[NUMERIC_COLS])
    return encoded


def pretty_name(col: str) -> str:
    """'Contract_Month-to-month' -> 'Contract: Month-to-month' etc."""
    names = {
        "tenure": "Short Tenure",
        "MonthlyCharges": "High Monthly Charges",
        "TotalCharges": "Total Charges",
        "SeniorCitizen": "Senior Citizen",
    }
    if col in names:
        return names[col]
    return col.replace("_", ": ", 1)


def top_risk_drivers(x: pd.DataFrame, n: int = 3):
    """
    Per-customer explanation for a linear model.
    contribution_i = coefficient_i * (scaled) feature value_i
    Positive contribution => pushes towards churn.
    """
    coefs = model.coef_[0]
    contrib = pd.Series(coefs * x.iloc[0].values, index=x.columns)
    return contrib.sort_values(ascending=False).head(n), contrib.sort_values().head(n)


# ----------------------------------------------------------------------------
# 6. GAUGE CHART
# ----------------------------------------------------------------------------
def gauge(prob: float) -> go.Figure:
    """Semi-circular gauge showing churn probability (0-100%)."""
    color = "#2ecc71" if prob < 0.35 else "#f39c12" if prob < RISK_THRESHOLD else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Churn Probability", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 35], "color": "rgba(46,204,113,0.25)"},
                {"range": [35, RISK_THRESHOLD * 100], "color": "rgba(243,156,18,0.25)"},
                {"range": [RISK_THRESHOLD * 100, 100], "color": "rgba(231,76,60,0.25)"},
            ],
            "threshold": {"line": {"color": "black", "width": 3},
                          "thickness": 0.8, "value": RISK_THRESHOLD * 100},
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ----------------------------------------------------------------------------
# 7. MAIN PAGE
# ----------------------------------------------------------------------------
st.title("📉 Customer Churn Prediction System")
st.markdown("Predict whether a telecom customer is likely to leave, "
            "and understand **why**, so retention teams can act early.")
st.divider()

if not submitted:
    # Landing state before the first prediction
    st.info("👈 Fill in the customer profile in the sidebar and press **Predict Churn**.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Logistic Regression")
    c2.metric("F1-Score", "63.4%")
    c3.metric("Imbalance Handling", "SMOTE")
    st.stop()

# ---- Run the prediction ------------------------------------------------------
X = build_feature_row()
churn_prob = float(model.predict_proba(X)[0][1])
is_high_risk = churn_prob >= RISK_THRESHOLD

# ---- Result banner + gauge ---------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Prediction Result")
    if is_high_risk:
        st.error("🚨 **High Risk of Churn**: this customer is likely to leave.")
    else:
        st.success("✅ **Customer is likely to stay.**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{churn_prob:.1%}")
    m2.metric("Retention Probability", f"{1 - churn_prob:.1%}")
    m3.metric("Risk Level",
              "High" if churn_prob >= RISK_THRESHOLD
              else "Medium" if churn_prob >= 0.35 else "Low")

    st.progress(min(max(churn_prob, 0.0), 1.0), text=f"Churn risk score: {churn_prob:.1%}")

    # Simple recommended action based on the result
    if is_high_risk:
        st.markdown("**Suggested action:** offer a contract upgrade or loyalty discount, "
                    "and schedule a proactive support call.")
    else:
        st.markdown("**Suggested action:** keep the customer engaged with regular "
                    "check-ins and cross-sell relevant services.")

with right:
    st.plotly_chart(gauge(churn_prob), use_container_width=True)

st.divider()

# ---- Business insights: top 3 drivers ---------------------------------------
st.subheader("🔎 Top 3 Features Driving This Customer's Risk")

if hasattr(model, "coef_"):
    risk_up, risk_down = top_risk_drivers(X)

    c_up, c_down = st.columns(2)
    with c_up:
        st.markdown("**Factors increasing churn risk**")
        for feat, val in risk_up.items():
            st.markdown(
                f'<div class="driver-card"><b>{pretty_name(feat)}</b><br>'
                f'<small>Impact on log-odds: +{val:.2f}</small></div>',
                unsafe_allow_html=True)
    with c_down:
        st.markdown("**Factors protecting against churn**")
        for feat, val in risk_down.items():
            st.markdown(
                f'<div class="driver-card safe"><b>{pretty_name(feat)}</b><br>'
                f'<small>Impact on log-odds: {val:.2f}</small></div>',
                unsafe_allow_html=True)
    st.caption("Impact = model coefficient x the customer's (scaled) feature value, "
               "computed directly from the Logistic Regression model.")
else:
    # Generic fallback if a non-linear model is loaded
    st.markdown("- 📆 **Short tenure**: newer customers churn more\n"
                "- 📝 **Month-to-month contract**: no long-term commitment\n"
                "- 💸 **High monthly charges**: price sensitivity")

# ---- Transparency: show what was fed to the model ---------------------------
with st.expander("🛠 View processed model input"):
    st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)
    st.caption("Numeric columns are standardised; categorical columns are one-hot encoded.")

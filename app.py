"""
Customer Churn Prediction System - Streamlit App
=================================================
Run with:   streamlit run app.py
Requires:   streamlit, pandas, numpy, scikit-learn, plotly, joblib
Files expected in the models/ folder:
    model.pkl   -> trained Logistic Regression (best F1 = 63.4%)
    scaler.pkl  -> StandardScaler fitted on [tenure, MonthlyCharges, TotalCharges]
Optional:
    columns.pkl -> list of the exact feature columns used in training
                   (only needed if your model has no `feature_names_in_`)
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

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
        .strength {font-size: 0.8rem; opacity: 0.75;}
        .bar {height: 6px; background: rgba(120,120,120,0.2); border-radius: 3px; margin-top: 6px;}
        .bar .fill {height: 6px; border-radius: 3px; background: #e74c3c;}
        .driver-card.safe .fill {background: #2ecc71;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent      # folder containing app.py
MODEL_DIR = BASE_DIR / "models"                 # saved model artifacts live here

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
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    # Work out the exact column order the model was trained on.
    if hasattr(model, "feature_names_in_"):          # set when trained on a DataFrame
        columns = list(model.feature_names_in_)
    elif (MODEL_DIR / "columns.pkl").exists():              # fallback: a saved list of columns
        columns = list(joblib.load(MODEL_DIR / "columns.pkl"))
    else:
        columns = None
    return model, scaler, columns


try:
    model, scaler, TRAIN_COLUMNS = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Could not find a required file: **{e.filename}**. "
             "Place `model.pkl` and `scaler.pkl` in the `models/` folder next to `app.py`.")
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
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
        total = st.number_input(
            "Total Charges ($)", 0.0, 10000.0,
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


# Plain-English names for each categorical question
FRIENDLY_GROUPS = {
    "gender": "Gender", "Partner": "Has partner", "Dependents": "Has dependents",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
}
MIN_EFFECT = 0.05   # factors weaker than this are hidden (they barely move the result)


def compute_factor_effects(x: pd.DataFrame):
    """
    Explain ONE prediction in plain English (valid for a linear model).
    Each factor gets an 'effect' on the customer's churn risk compared with an
    average customer. Positive = raises risk, negative = lowers risk.

    - Numeric features: coefficient x scaled value (scaled value 0 = average customer)
    - Categorical features: the chosen option's coefficient minus the average over
      all options of that question. (Because one option per question is dropped
      during training, this lets e.g. 'Month-to-month' show up properly.)
    """
    coefs = pd.Series(model.coef_[0], index=x.columns)
    scaled = x.iloc[0]
    effects = []   # list of (label, effect)

    # ---- numeric factors (direction decided by the customer's actual value) ----
    if "tenure" in coefs.index:
        # Tenure and total charges move together (long-time customers have paid more),
        # so their effects are combined into ONE easy-to-read "customer history" factor.
        history = coefs["tenure"] * scaled["tenure"]
        if "TotalCharges" in coefs.index:
            history += coefs["TotalCharges"] * scaled["TotalCharges"]
        word = "Short" if scaled["tenure"] < 0 else "Long"
        effects.append((f"{word} tenure ({tenure} months)", history))
    if "MonthlyCharges" in coefs.index:
        word = "Higher" if scaled["MonthlyCharges"] > 0 else "Lower"
        effects.append((f"{word}-than-average monthly bill (${monthly:,.0f})",
                        coefs["MonthlyCharges"] * scaled["MonthlyCharges"]))
    if "SeniorCitizen" in coefs.index and senior == "Yes":
        effects.append(("Senior citizen", coefs["SeniorCitizen"]))

    # ---- categorical factors ----
    selections = {
        "gender": gender, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    }
    for group, value in selections.items():
        group_cols = [c for c in coefs.index if c.startswith(group + "_")]
        if not group_cols:
            continue
        chosen = coefs.get(f"{group}_{value}", 0.0)           # 0 = the dropped baseline option
        average = coefs[group_cols].sum() / (len(group_cols) + 1)
        effects.append((f"{FRIENDLY_GROUPS[group]}: {value}", chosen - average))

    return effects


def strength_of(effect: float):
    """Turn a number into words and a bar width a non-technical user can read."""
    size = abs(effect)
    if size >= 0.40:
        return "Strong influence", 100
    if size >= 0.15:
        return "Moderate influence", 60
    return "Slight influence", 30


def driver_card(label: str, effect: float, safe: bool) -> str:
    """HTML card: factor name + influence label + small bar."""
    text, width = strength_of(effect)
    css = "driver-card safe" if safe else "driver-card"
    return (f'<div class="{css}"><b>{label}</b><br>'
            f'<span class="strength">{text}</span>'
            f'<div class="bar"><div class="fill" style="width:{width}%"></div></div></div>')


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
st.subheader("🔎 Why This Prediction? Top Factors")

if hasattr(model, "coef_"):
    effects = [(lbl, e) for lbl, e in compute_factor_effects(X) if abs(e) >= MIN_EFFECT]
    risk_up = sorted([t for t in effects if t[1] > 0], key=lambda t: -t[1])[:3]
    risk_down = sorted([t for t in effects if t[1] < 0], key=lambda t: t[1])[:3]

    # One-sentence summary in plain English
    if risk_up:
        reasons = "; ".join(lbl.lower() for lbl, _ in risk_up)
        st.markdown(f"**In short:** this customer's risk is mainly pushed up by {reasons}.")
    else:
        st.markdown("**In short:** no single factor stands out as raising this customer's risk.")

    c_up, c_down = st.columns(2)
    with c_up:
        st.markdown("**⬆️ Raising the risk of leaving**")
        if risk_up:
            for lbl, e in risk_up:
                st.markdown(driver_card(lbl, e, safe=False), unsafe_allow_html=True)
        else:
            st.caption("None of the factors raise the risk noticeably.")
    with c_down:
        st.markdown("**⬇️ Helping keep the customer**")
        if risk_down:
            for lbl, e in risk_down:
                st.markdown(driver_card(lbl, e, safe=True), unsafe_allow_html=True)
        else:
            st.caption("None of the factors protect against leaving noticeably.")

    st.caption("Each bar shows how strongly a factor pushes this customer's risk up or down "
               "compared with an average customer, based on what the model learned from past data.")
else:
    # Generic fallback if a non-linear model is loaded
    st.markdown("- 📆 **Short tenure**: newer customers churn more\n"
                "- 📝 **Month-to-month contract**: no long-term commitment\n"
                "- 💸 **High monthly charges**: price sensitivity")

# ---- Transparency: show what was fed to the model ---------------------------
with st.expander("🛠 View processed model input"):
    st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)
    st.caption("Numeric columns are standardised; categorical columns are one-hot encoded.")

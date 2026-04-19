import streamlit as st
import pickle
import pandas as pd
import sys
import os

# ─────────────────────────────────────────
# PATH SETUP — works both locally and on Streamlit Cloud
# ─────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_engineering import build_features

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# LOAD MODEL (cached so it loads only once)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "models", "full_pipeline.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "models", "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# ─────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────
def predict(income, credit, annuity, days_birth, days_employed, family, ext1, ext2, ext3):
    data = pd.DataFrame([{
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT":       credit,
        "AMT_ANNUITY":      annuity,
        "DAYS_BIRTH":       days_birth,
        "DAYS_EMPLOYED":    days_employed,
        "CNT_FAM_MEMBERS":  family,
        "EXT_SOURCE_1":     ext1,
        "EXT_SOURCE_2":     ext2,
        "EXT_SOURCE_3":     ext3,
    }])
    data = build_features(data)
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_columns]
    prob = model.predict_proba(data)[0][1]

    if prob < 0.3:
        risk = "LOW"
    elif prob < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    decision = "REJECT" if prob >= 0.50 else "APPROVE"

    reasons = []
    row = data.iloc[0]
    if row.get("CREDIT_TO_INCOME_RATIO", 0) > 0.5:
        reasons.append("High credit relative to income")
    if row.get("EXT_SOURCE_MEAN", 1) < 0.4:
        reasons.append("Low external credit score")
    if not reasons:
        reasons.append("No major risk signals")

    return round(float(prob), 4), decision, risk, reasons

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
.stApp { background-color: #0d1117; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }

.header-block {
    border-left: 4px solid #238636;
    padding: 12px 20px;
    margin-bottom: 32px;
    background: #161b22;
    border-radius: 0 8px 8px 0;
}
.header-block h1 {
    font-size: 1.6rem; margin: 0;
    color: #58a6ff; font-family: 'IBM Plex Mono', monospace;
}
.header-block p { margin: 4px 0 0 0; color: #8b949e; font-size: 0.85rem; }

.input-section {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
    color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 12px;
}
.result-approve {
    background: #0f2a1a; border: 1px solid #238636;
    border-radius: 8px; padding: 24px; text-align: center;
}
.result-reject {
    background: #2a0f0f; border: 1px solid #da3633;
    border-radius: 8px; padding: 24px; text-align: center;
}
.result-decision { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; margin: 0; }
.result-approve .result-decision { color: #3fb950; }
.result-reject  .result-decision { color: #f85149; }
.result-prob { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; color: #8b949e; margin-top: 6px; }

.risk-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; margin-top: 10px;
}
.risk-LOW    { background: #0f2a1a; color: #3fb950; border: 1px solid #238636; }
.risk-MEDIUM { background: #2a1f0f; color: #d29922; border: 1px solid #9e6a03; }
.risk-HIGH   { background: #2a0f0f; color: #f85149; border: 1px solid #da3633; }

.driver-item {
    background: #1c2128; border: 1px solid #30363d;
    border-radius: 6px; padding: 10px 14px;
    margin-bottom: 8px; font-size: 0.88rem; color: #c9d1d9;
}
.driver-item::before { content: "▸ "; color: #58a6ff; font-family: 'IBM Plex Mono', monospace; }

.prob-bar-container {
    background: #21262d; border-radius: 4px;
    height: 8px; margin: 12px 0; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 4px; }

.stButton > button {
    background: #238636 !important; color: #ffffff !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    padding: 10px 28px !important; width: 100% !important;
}
label { color: #8b949e !important; font-size: 0.85rem !important; }

.footnote {
    color: #484f58; font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 40px; border-top: 1px solid #21262d; padding-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🏦 Credit Risk Scorer</h1>
    <p>Home Credit Default Risk · LightGBM · AUC 0.783 · Business-Optimized Threshold</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Financial Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        income  = st.number_input("Annual Income (₹)", min_value=10000.0, max_value=10000000.0, value=202500.0, step=5000.0)
        credit  = st.number_input("Credit Amount (₹)", min_value=10000.0, max_value=5000000.0,  value=406597.0, step=5000.0)
    with c2:
        annuity = st.number_input("Annuity Amount (₹)", min_value=1000.0, max_value=500000.0, value=24700.0, step=1000.0)
        family  = st.number_input("Family Members", min_value=1, max_value=20, value=2, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        age_years     = st.number_input("Age (years)", min_value=18, max_value=70, value=33, step=1)
        days_birth    = -age_years * 365
    with c4:
        emp_years     = st.number_input("Employment Duration (years)", min_value=0, max_value=40, value=3, step=1)
        days_employed = -emp_years * 365
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">External Credit Scores (0 = worst · 1 = best)</div>', unsafe_allow_html=True)
    ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.45, 0.01)
    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.26, 0.01)
    ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.50, 0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.button("▶  Score Applicant")

with right:
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div style="background:#161b22; border:1px dashed #30363d; border-radius:8px;
                    padding:40px 24px; text-align:center; color:#484f58;">
            <div style="font-size:2rem; margin-bottom:12px;">⬅</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem;">
                Fill in applicant details<br>and click Score Applicant
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            prob, decision, risk, drivers = predict(
                income, credit, annuity,
                float(days_birth), float(days_employed),
                float(family), ext1, ext2, ext3
            )

            bar_color = "#3fb950" if prob < 0.3 else "#d29922" if prob < 0.6 else "#f85149"
            css_class = "result-approve" if decision == "APPROVE" else "result-reject"
            icon      = "✅" if decision == "APPROVE" else "❌"

            st.markdown(f"""
            <div class="{css_class}">
                <p class="result-decision">{icon} {decision}</p>
                <p class="result-prob">Default probability: <strong>{prob:.1%}</strong></p>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:{bar_color};"></div>
                </div>
                <span class="risk-badge risk-{risk}">RISK: {risk}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Key Risk Drivers</div>', unsafe_allow_html=True)
            for d in drivers:
                st.markdown(f'<div class="driver-item">{d}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:8px;
                        padding:14px 18px; font-family:'IBM Plex Mono',monospace;
                        font-size:0.78rem; color:#8b949e; line-height:1.8;">
                Income &nbsp;&nbsp;&nbsp; ₹{income:,.0f}<br>
                Credit &nbsp;&nbsp;&nbsp; ₹{credit:,.0f}<br>
                Annuity &nbsp;&nbsp; ₹{annuity:,.0f}<br>
                Age &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {age_years} yrs &nbsp;|&nbsp; Employment {emp_years} yrs<br>
                EXT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {ext1:.2f} / {ext2:.2f} / {ext3:.2f}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footnote">
    Home Credit Default Risk · LightGBM (AUC 0.783, KS 42.7%) ·
    Threshold optimized at 33:1 FN/FP cost ratio · Net value ₹93.9 crore ·
    github.com/kaushalkumarma2025/home-credit-default-risk
</div>
""", unsafe_allow_html=True)

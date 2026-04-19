import streamlit as st
import requests
import json

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

.stApp {
    background-color: #0d1117;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #e6edf3;
}

.header-block {
    border-left: 4px solid #238636;
    padding: 12px 20px;
    margin-bottom: 32px;
    background: #161b22;
    border-radius: 0 8px 8px 0;
}

.header-block h1 {
    font-size: 1.6rem;
    margin: 0;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}

.header-block p {
    margin: 4px 0 0 0;
    color: #8b949e;
    font-size: 0.85rem;
}

.input-section {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

.result-approve {
    background: #0f2a1a;
    border: 1px solid #238636;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
}

.result-reject {
    background: #2a0f0f;
    border: 1px solid #da3633;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
}

.result-decision {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
}

.result-approve .result-decision { color: #3fb950; }
.result-reject .result-decision  { color: #f85149; }

.result-prob {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: #8b949e;
    margin-top: 6px;
}

.risk-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    margin-top: 10px;
}

.risk-LOW    { background: #0f2a1a; color: #3fb950; border: 1px solid #238636; }
.risk-MEDIUM { background: #2a1f0f; color: #d29922; border: 1px solid #9e6a03; }
.risk-HIGH   { background: #2a0f0f; color: #f85149; border: 1px solid #da3633; }

.driver-item {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #c9d1d9;
}

.driver-item::before {
    content: "▸ ";
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}

.prob-bar-container {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    margin: 12px 0;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.api-error {
    background: #2a1a0f;
    border: 1px solid #9e6a03;
    border-radius: 8px;
    padding: 16px;
    color: #d29922;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background: #2ea043 !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSlider"] {
    background: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

label {
    color: #8b949e !important;
    font-size: 0.85rem !important;
}

.stSlider [data-baseweb="slider"] {
    color: #58a6ff;
}

hr {
    border-color: #30363d;
}

.footnote {
    color: #484f58;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 40px;
    border-top: 1px solid #21262d;
    padding-top: 16px;
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
# API CONFIG
# ─────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ─────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    # --- Financial Info ---
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Financial Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Annual Income (₹)", min_value=10000.0, max_value=10000000.0,
                                  value=202500.0, step=5000.0)
        credit = st.number_input("Credit Amount (₹)", min_value=10000.0, max_value=5000000.0,
                                  value=406597.0, step=5000.0)
    with c2:
        annuity = st.number_input("Annuity Amount (₹)", min_value=1000.0, max_value=500000.0,
                                   value=24700.0, step=1000.0)
        family  = st.number_input("Family Members", min_value=1, max_value=20, value=2, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Personal Info ---
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        age_years     = st.number_input("Age (years)", min_value=18, max_value=70, value=33, step=1)
        days_birth    = -age_years * 365
    with c4:
        emp_years     = st.number_input("Employment Duration (years)", min_value=0, max_value=40,
                                         value=3, step=1)
        days_employed = -emp_years * 365
    st.markdown('</div>', unsafe_allow_html=True)

    # --- External Credit Scores ---
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">External Credit Scores (0 = worst · 1 = best)</div>',
                unsafe_allow_html=True)
    ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.45, 0.01)
    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.26, 0.01)
    ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.50, 0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.button("▶  Score Applicant")

# ─────────────────────────────────────────
# RIGHT PANEL — RESULTS
# ─────────────────────────────────────────
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
        payload = {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT":       credit,
            "AMT_ANNUITY":      annuity,
            "DAYS_BIRTH":       float(days_birth),
            "DAYS_EMPLOYED":    float(days_employed),
            "CNT_FAM_MEMBERS":  float(family),
            "EXT_SOURCE_1":     ext1,
            "EXT_SOURCE_2":     ext2,
            "EXT_SOURCE_3":     ext3,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            prob      = result["default_probability"]
            decision  = result["decision"]
            risk      = result["risk_level"]
            drivers   = result["key_drivers"]

            # colour for prob bar
            if prob < 0.3:
                bar_color = "#3fb950"
            elif prob < 0.6:
                bar_color = "#d29922"
            else:
                bar_color = "#f85149"

            css_class = "result-approve" if decision == "APPROVE" else "result-reject"
            icon      = "✅" if decision == "APPROVE" else "❌"

            st.markdown(f"""
            <div class="{css_class}">
                <p class="result-decision">{icon} {decision}</p>
                <p class="result-prob">Default probability: <strong>{prob:.1%}</strong></p>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill"
                         style="width:{prob*100:.1f}%; background:{bar_color};"></div>
                </div>
                <span class="risk-badge risk-{risk}">RISK: {risk}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Key Risk Drivers</div>', unsafe_allow_html=True)
            for d in drivers:
                st.markdown(f'<div class="driver-item">{d}</div>', unsafe_allow_html=True)

            # input summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:8px;
                        padding:14px 18px; font-family:'IBM Plex Mono',monospace;
                        font-size:0.78rem; color:#8b949e; line-height:1.8;">
                Income &nbsp;&nbsp;&nbsp; ₹{income:,.0f}<br>
                Credit &nbsp;&nbsp;&nbsp; ₹{credit:,.0f}<br>
                Annuity &nbsp;&nbsp; ₹{annuity:,.0f}<br>
                Age &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {age_years} yrs &nbsp;|&nbsp;
                Employment {emp_years} yrs<br>
                EXT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {ext1:.2f} / {ext2:.2f} / {ext3:.2f}
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.markdown("""
            <div class="api-error">
                ⚠ Cannot connect to API at http://127.0.0.1:8000<br><br>
                Start the FastAPI server first:<br><br>
                <code>cd C:\\Users\\Admin\\home-credit-default-risk\\api</code><br>
                <code>uvicorn app:app --reload</code>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="api-error">
                ⚠ Error: {str(e)}
            </div>
            """, unsafe_allow_html=True)

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

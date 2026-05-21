"""
app.py  -  FNB DataQuest 2026 | Credit Intelligence Platform
Moloi Qolani Truelove & Tshegofatso Tshepang Chikwane
Sol Plaatje University

Run:  streamlit run app.py

API KEY SETUP:
  Local  → put your key in .streamlit/secrets.toml  (see that file)
  Cloud  → Streamlit dashboard → Settings → Secrets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys, base64
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from model_engine import (
    train_model, load_and_clean, predict_applicant,
    calculate_iv, woe_transform,
    NAVY, TEAL, GOLD, RED, GREEN, MUTED, PALE
)
from report_generator import generate_approved_report, generate_declined_report


# ── GROQ API KEY  ──────────────────────────────────────────────────────────────
# Reads from .streamlit/secrets.toml locally, or from Streamlit Cloud secrets.
# Falls back to a clear error message so the app still loads without crashing.
def _get_groq_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return ""


# PAGE CONFIG
st.set_page_config(
    page_title="FNB DataQuest 2026 | Credit Intelligence",
    page_icon="https://www.fnb.co.za/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
# All text colours are declared explicitly so the UI looks correct on both
# light browsers and browsers/OS set to dark-mode.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Force light-mode base so dark-OS-theme can't override our palette ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"], .main, .block-container {
    background-color: #F4F6F9 !important;
    color: #0A1628 !important;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #0A1628;
}

/* ── All generic Streamlit text ── */
p, span, label, div, li, td, th, h1, h2, h3, h4, h5, h6 {
    color: #0A1628;
}

/* ── Streamlit widget labels ── */
[data-testid="stWidgetLabel"] > div,
[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stNumberInput label,
.stTextInput label, .stCheckbox label,
.stRadio label, .stSlider label,
.stTextArea label {
    color: #0A1628 !important;
}

/* ── Streamlit markdown / caption ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th,
[data-testid="stCaptionContainer"] p {
    color: #0A1628 !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
.dataframe td, .dataframe th {
    color: #0A1628 !important;
    background-color: #ffffff !important;
}

/* ── Input fields ── */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    color: #0A1628 !important;
    background-color: #ffffff !important;
    border-color: #c6dde1 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #0A1628 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #007B8A !important;
    border-bottom-color: #007B8A !important;
}

/* ── Metric widget ── */
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #0A1628 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #0d2040 100%) !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: white !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
    display: block;
    margin: 2px 0;
    font-size: 14px !important;
    transition: background 0.2s;
    color: white !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.1);
}

/* ── Custom card components ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #E8E8E8;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
}
.metric-card .metric-label {
    font-size: 10px;
    color: #888888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-card .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #0A1628;
}
.metric-card .metric-sub {
    font-size: 11px;
    color: #007B8A;
    margin-top: 4px;
}

.chapter-header {
    background: linear-gradient(135deg, #0A1628 0%, #007B8A 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 28px;
}
.chapter-header h1 {
    color: #ffffff !important;
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    margin: 0 0 6px 0;
}
.chapter-header p {
    color: rgba(232,244,246,0.80) !important;
    font-size: 14px;
    margin: 0;
}

.decision-approved {
    background: #D6F0E0;
    color: #1A6B3C !important;
    border: 2px solid #1A6B3C;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-size: 28px;
    font-weight: 700;
}
.decision-declined {
    background: #FDECEA;
    color: #C0392B !important;
    border: 2px solid #C0392B;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-size: 28px;
    font-weight: 700;
}

.chat-user {
    background: #0A1628;
    color: #ffffff !important;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 78%;
    margin-left: auto;
    font-size: 14px;
}
.chat-bot {
    background: #E8F4F6;
    color: #0A1628 !important;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 82%;
    font-size: 14px;
    line-height: 1.6;
}

.score-ring {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.20);
}

/* ── Streamlit submit/form buttons ── */
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"],
.stButton button {
    color: #0A1628 !important;
    background-color: #ffffff !important;
    border-color: #007B8A !important;
}
[data-testid="baseButton-primary"]:hover,
.stButton button:hover {
    background-color: #007B8A !important;
    color: #ffffff !important;
}

/* ── Hide chrome ── */
[data-testid="collapsedControl"]                                             { display: none !important; }
button[kind="header"]                                                        { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > div:first-child button { display: none !important; }
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# CACHE MODEL & DATA
@st.cache_resource(show_spinner="Training credit model - please wait...")
def get_model():
    return train_model()

@st.cache_resource
def get_data():
    return load_and_clean()


# ── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 24px;">
        <div style="color:#C9A84C; font-size:11px; letter-spacing:3px; text-transform:uppercase; margin-bottom:8px;">FNB DataQuest 2026</div>
        <div style="color:white; font-size:20px; font-weight:700; line-height:1.3;">Credit Intelligence<br>Platform</div>
        <div style="color:rgba(232,244,246,0.55); font-size:11px; margin-top:8px;">Moloi & Chikwane - SPU</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.12); margin-bottom:20px;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Data Quality",
            "EDA Explorer",
            "Model & Scorecard",
            "Business Dashboard",
            "Credit AI Chatbot",
            "Loan Decision Report",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.12); margin:20px 0 16px;">
    <div style="font-size:11px; color:rgba(232,244,246,0.40); line-height:1.8; padding: 0 4px;">
        Model: Logistic Regression (WoE)<br>
        Dataset: 120,960 applications<br>
        Default Rate: 15.4%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.12); margin:20px 0 16px;">
    <div style="color:#C9A84C; font-size:10px; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px; padding: 0 4px;">
        🤖 Navigator — System Guide
    </div>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as _nav_comp
    _nav_comp.html("""<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'DM Sans',sans-serif;background:transparent;}

  #nav-toggle {
    width:100%; padding:10px 14px; border-radius:10px;
    background:linear-gradient(135deg,#007B8A,#0A1628);
    border:1.5px solid rgba(201,168,76,0.6);
    color:white; font-size:13px; font-weight:600;
    cursor:pointer; display:flex; align-items:center; gap:8px;
    transition:opacity 0.2s;
  }
  #nav-toggle:hover{opacity:0.88;}
  #nav-toggle svg{flex-shrink:0;}

  #nav-panel {
    display:none; flex-direction:column;
    background:#ffffff; border-radius:12px;
    border:1px solid #e0eff2;
    box-shadow:0 8px 32px rgba(10,22,40,0.18);
    overflow:hidden; margin-top:8px;
    max-height:480px;
  }
  #nav-panel.open{display:flex;}

  #nav-messages {
    flex:1; overflow-y:auto; padding:12px;
    display:flex; flex-direction:column; gap:8px;
    max-height:300px;
    scrollbar-width:thin; scrollbar-color:#c0d8de transparent;
  }
  .nav-msg{padding:9px 12px;border-radius:12px;font-size:12px;line-height:1.5;}
  .nav-msg.bot{background:#EBF5F7;color:#0A1628;border-bottom-left-radius:3px;}
  .nav-msg.user{background:#0A1628;color:white;border-bottom-right-radius:3px;text-align:right;}
  .nav-msg.typing{color:#888;font-style:italic;}

  .nav-chips{display:flex;flex-wrap:wrap;gap:5px;padding:6px 12px 10px;}
  .nav-chip{
    background:white;border:1.5px solid #007B8A;border-radius:16px;
    padding:4px 10px;font-size:11px;color:#007B8A;cursor:pointer;
    transition:background 0.15s,color 0.15s;white-space:nowrap;
  }
  .nav-chip:hover{background:#007B8A;color:white;}

  #nav-input-row{
    border-top:1px solid #e8eff2;padding:8px 10px;
    display:flex;align-items:center;gap:6px;background:#f8fcfd;
  }
  #nav-input{
    flex:1;border:1px solid #c6dde1;border-radius:16px;
    padding:7px 12px;font-size:12px;outline:none;color:#0A1628;background:white;
  }
  #nav-input:focus{border-color:#007B8A;}
  #nav-send{
    width:30px;height:30px;background:#007B8A;border:none;
    border-radius:50%;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;
  }
  #nav-send:hover{background:#0A1628;}
  #nav-send svg{pointer-events:none;}
</style>
</head>
<body>

<button id="nav-toggle" onclick="navOpen()">
  <svg width="18" height="18" viewBox="0 0 28 28" fill="none">
    <rect x="5" y="8" width="18" height="13" rx="3" fill="white" opacity="0.9"/>
    <circle cx="10.5" cy="13.5" r="2" fill="#007B8A"/>
    <circle cx="17.5" cy="13.5" r="2" fill="#007B8A"/>
    <circle cx="11.2" cy="13" r="0.6" fill="white"/>
    <circle cx="18.2" cy="13" r="0.6" fill="white"/>
    <rect x="10" y="17" width="8" height="1.5" rx="0.75" fill="#C9A84C"/>
    <line x1="14" y1="8" x2="14" y2="4.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="14" cy="3.5" r="1.2" fill="#C9A84C"/>
    <rect x="2.5" y="11" width="2.5" height="5" rx="1.25" fill="white" opacity="0.7"/>
    <rect x="23" y="11" width="2.5" height="5" rx="1.25" fill="white" opacity="0.7"/>
  </svg>
  Ask Navigator
</button>

<div id="nav-panel">
  <div id="nav-messages">
    <div class="nav-msg bot">
      👋 Hi! I'm <b>Navigator</b>. Ask me about any term or page in this platform, or tap a chip below!<br><br>
      For loan &amp; finance questions, use the <b>Credit AI Chatbot</b> page.
    </div>
  </div>
  <div class="nav-chips" id="nav-chips">
    <div class="nav-chip" onclick="navAsk('What is DTI?')">DTI?</div>
    <div class="nav-chip" onclick="navAsk('What is WoE?')">WoE?</div>
    <div class="nav-chip" onclick="navAsk('What is AUC?')">AUC?</div>
    <div class="nav-chip" onclick="navAsk('What is IV?')">IV?</div>
    <div class="nav-chip" onclick="navAsk('What is Gini?')">Gini?</div>
    <div class="nav-chip" onclick="navAsk('What is KS?')">KS?</div>
    <div class="nav-chip" onclick="navAsk('What is a threshold?')">Threshold?</div>
    <div class="nav-chip" onclick="navAsk('Why logistic regression?')">Why logistic?</div>
    <div class="nav-chip" onclick="navAsk('What does each page do?')">Pages?</div>
    <div class="nav-chip" onclick="navAsk('What is a credit score?')">Score?</div>
    <div class="nav-chip" onclick="navAsk('What is a confusion matrix?')">Confusion matrix?</div>
    <div class="nav-chip" onclick="navAsk('What is ROC curve?')">ROC?</div>
  </div>
  <div id="nav-input-row">
    <input id="nav-input" placeholder="Ask anything..." onkeydown="if(event.key==='Enter') navSend()"/>
    <button id="nav-send" onclick="navSend()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
</div>

<script>
var KB={
  dti:{kw:["dti","debt to income","debt-to-income"],ans:"<b>DTI - Debt-to-Income Ratio</b><br><br>How much of your monthly income goes toward debt.<br><br>Formula: Total Monthly Debt / Gross Monthly Income<br><br>E.g. earn R10,000/month, pay R3,000 = DTI 0.30 (30%). Above <b>0.35</b> is high risk in this model."},
  woe:{kw:["woe","weight of evidence"],ans:"<b>WoE - Weight of Evidence</b><br><br>Transforms each feature into a number predicting default risk.<br><br>Formula: ln(% Good / % Bad)<br><br>Positive WoE = lower risk. Negative WoE = higher risk.<br><br>See it on <b>EDA Explorer - Univariate</b>."},
  iv:{kw:["iv","information value"],ans:"<b>IV - Information Value</b><br><br>Measures how powerful a feature is at predicting default.<br><br>Less than 0.02 = Useless | 0.02-0.10 = Weak | 0.10-0.30 = Medium | 0.30-0.50 = Strong | Above 0.50 = Very strong<br><br>See rankings on <b>EDA Explorer - WoE/IV Rankings</b>."},
  auc:{kw:["auc","area under curve"],ans:"<b>AUC - Area Under the ROC Curve</b><br><br>Measures how well the model separates defaulters from good customers (0 to 1).<br><br>0.50 = random | 0.70-0.80 = good | 1.00 = perfect<br><br>Our model scores <b>0.7984</b>, beating the baseline of 0.68."},
  gini:{kw:["gini"],ans:"<b>Gini Coefficient</b><br><br>Gini = 2 x AUC - 1. Skill above random guessing.<br><br>Less than 30% = Poor | 30-50% = Acceptable | 50-70% = Good | Above 70% = Excellent"},
  ks:{kw:["ks","kolmogorov"],ans:"<b>KS - Kolmogorov-Smirnov Statistic</b><br><br>Max separation between good customers and defaulters score distributions.<br><br>Less than 20% = Poor | 20-40% = Acceptable | 40-60% = Good | Above 60% = Very strong"},
  threshold:{kw:["threshold","cutoff","cut-off"],ans:"<b>Threshold - Approval Cut-off</b><br><br>The line between APPROVED and DECLINED.<br><br>Probability below threshold = APPROVED<br>Probability above threshold = DECLINED<br><br>Default is 0.50. Adjust live on <b>Business Dashboard</b>."},
  logistic:{kw:["logistic","why logistic"],ans:"<b>Why Logistic Regression?</b><br><br>Banks require explainability by law.<br><br>Every decision can be explained to a customer. Compliant with Basel III/IV and National Credit Act. Auditable by NCR and SARB.<br><br>LightGBM scores AUC 0.82 but is a black box - not usable in a real bank."},
  roc:{kw:["roc","roc curve"],ans:"<b>ROC Curve</b><br><br>Shows the trade-off between catching defaulters vs wrongly declining good customers.<br><br>X-axis: False Positive Rate | Y-axis: True Positive Rate<br><br>Further top-left = better model."},
  confusion:{kw:["confusion matrix","confusion"],ans:"<b>Confusion Matrix</b><br><br>4 possible credit decision outcomes:<br><br>Good customer correctly approved<br>Good customer wrongly declined (lost revenue)<br>Defaulter wrongly approved (bank loses money)<br>Defaulter correctly declined<br><br>See it on <b>Model and Scorecard - Model Performance</b>."},
  score:{kw:["credit score","scorecard","score range"],ans:"<b>Credit Score (300-900)</b><br><br>300-449 = Very High Risk<br>450-579 = High Risk<br>580-649 = Medium Risk<br>650-900 = Low Risk - strong approval candidate<br><br>See breakdown on <b>Model and Scorecard - Credit Scorecard</b>."},
  pages:{kw:["page","pages","each page","guide"],ans:"<b>Platform Pages</b><br><br>Home - Overview and live metrics<br>Data Quality - Missing values and outliers<br>EDA Explorer - Features and WoE bins<br>Model and Scorecard - Scorecard and ROC<br>Business Dashboard - Threshold impact<br>Credit AI Chatbot - Loan questions<br>Loan Decision Report - Score any applicant"},
  default_:{kw:["default","defaulted","not pay"],ans:"<b>Default</b><br><br>When a borrower fails to repay after missing payments.<br><br>0 = Did not default (good customer)<br>1 = Defaulted (bad customer)<br><br>About 15.4% of applicants defaulted. This is what the model predicts."}
};

var isOpen = false;

function navOpen(){
  isOpen = !isOpen;
  var panel = document.getElementById('nav-panel');
  var btn = document.getElementById('nav-toggle');
  if(isOpen){
    panel.classList.add('open');
    btn.innerHTML = btn.innerHTML.replace('Ask Navigator','Close Navigator');
    document.getElementById('nav-input').focus();
  } else {
    panel.classList.remove('open');
    btn.innerHTML = btn.innerHTML.replace('Close Navigator','Ask Navigator');
  }
}

function navAddMessage(text, role){
  var msgs = document.getElementById('nav-messages');
  var div = document.createElement('div');
  div.className = 'nav-msg ' + role;
  div.innerHTML = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function navRespond(q){
  q = q.toLowerCase().trim();
  for(var k in KB) if(KB[k].kw.some(function(kw){return q.indexOf(kw)>-1;})) return KB[k].ans;
  return 'That is outside my platform knowledge. For loan advice and personal finance, use the <b>Credit AI Chatbot</b> page in the sidebar!';
}

function navAsk(question){
  var chips = document.getElementById('nav-chips');
  if(chips) chips.style.display = 'none';
  navAddMessage(question, 'user');
  var typing = navAddMessage('Thinking...', 'bot typing');
  setTimeout(function(){
    typing.remove();
    navAddMessage(navRespond(question), 'bot');
  }, 300);
}

function navSend(){
  var input = document.getElementById('nav-input');
  var q = input.value.trim();
  if(!q) return;
  input.value = '';
  navAsk(q);
}
</script>
</body>
</html>""", height=520, scrolling=False)


# ── PAGE: HOME ─────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown("""
    <div class="chapter-header">
        <h1>FNB DataQuest 2026 - Credit Intelligence Platform</h1>
        <p>From Roots to Rise &nbsp;|&nbsp; Interpretable Credit Modelling &nbsp;|&nbsp; Sol Plaatje University</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### What is this platform?

        This is a complete credit risk intelligence system built for the FNB DataQuest 2026 competition.
        It demonstrates that logistic regression, when powered by professional WoE/IV feature engineering,
        can approach the performance of unconstrained machine learning models while remaining fully
        interpretable, auditable, and regulatorily compliant.

        #### Navigation Guide

        | Page | Purpose |
        |------|---------|
        | Data Quality | Inspect missing values, outliers, class imbalance |
        | EDA Explorer | Univariate and bivariate analysis, WoE/IV rankings |
        | Business Dashboard | Threshold analysis, Rand-value impact, Gini/KS metrics |
        | Credit AI Chatbot | Ask questions about the model in plain English |
        | Loan Decision Report | Score any applicant and generate a full decision report |
        """)

    with col2:
        md = get_model()
        st.markdown(f"""
        <div style="background:#0A1628; border-radius:14px; padding:24px; margin-top:8px;">
            <div style="color:#C9A84C; font-size:10px; letter-spacing:2px; text-transform:uppercase; margin-bottom:16px;">Live Model Metrics</div>
            <div style="margin-bottom:14px;">
                <div style="color:rgba(232,244,246,0.55); font-size:11px;">AUC Score</div>
                <div style="color:#ffffff; font-size:28px; font-weight:700;">{md['auc']:.4f}</div>
            </div>
            <div style="margin-bottom:14px;">
                <div style="color:rgba(232,244,246,0.55); font-size:11px;">Gini Coefficient</div>
                <div style="color:#C9A84C; font-size:24px; font-weight:700;">{md['gini']:.4f}</div>
            </div>
            <div style="margin-bottom:14px;">
                <div style="color:rgba(232,244,246,0.55); font-size:11px;">KS Statistic</div>
                <div style="color:#007B8A; font-size:24px; font-weight:700;">{md['ks']:.4f}</div>
            </div>
            <div>
                <div style="color:rgba(232,244,246,0.55); font-size:11px;">Gap Closed vs Baseline</div>
                <div style="color:#1A6B3C; font-size:24px; font-weight:700;">{(md['auc']-0.68)/(0.82-0.68)*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Team")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#E8F4F6; border-radius:10px; padding:18px;">
            <div style="color:#007B8A; font-size:10px; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px;">Author 1</div>
            <div style="font-size:18px; font-weight:700; color:#0A1628;">Moloi Qolani Truelove</div>
            <div style="font-size:12px; color:#555555; margin-top:4px;">Sol Plaatje University<br>Advanced Diploma in ICT: Applications Development (NQF 7)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#E8F4F6; border-radius:10px; padding:18px;">
            <div style="color:#007B8A; font-size:10px; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px;">Author 2</div>
            <div style="font-size:18px; font-weight:700; color:#0A1628;">Tshegofatso Tshepang Chikwane</div>
            <div style="font-size:12px; color:#555555; margin-top:4px;">Sol Plaatje University<br>BSc Data Science (NQF 7)</div>
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: DATA QUALITY ─────────────────────────────────────────────────────────
elif page == "Data Quality":
    st.markdown("""<div class="chapter-header"><h1>Data Quality Report</h1>
    <p>Missing values &nbsp;|&nbsp; Outliers &nbsp;|&nbsp; Class imbalance &nbsp;|&nbsp; Column overview</p></div>""",
    unsafe_allow_html=True)

    df, df_train, df_test = get_data()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Records</div>
        <div class="metric-value">{len(df):,}</div><div class="metric-sub">loan applications</div></div>""",
        unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Default Rate</div>
        <div class="metric-value" style="color:{RED};">{df['default_flag'].mean()*100:.1f}%</div>
        <div class="metric-sub">Class imbalance present</div></div>""", unsafe_allow_html=True)
    with c3:
        n_miss = (df.isnull().sum() > 0).sum()
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Cols with Missing</div>
        <div class="metric-value" style="color:{GOLD};">{n_miss}</div>
        <div class="metric-sub">of {df.shape[1]} total columns</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Features</div>
        <div class="metric-value">{df.shape[1]}</div><div class="metric-sub">raw columns</div></div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Values by Column")
        miss = df.isnull().sum() / len(df) * 100
        miss = miss[miss > 0].sort_values(ascending=True)
        colors = [RED if v > 40 else GOLD if v > 5 else TEAL for v in miss.values]
        fig = go.Figure(go.Bar(
            x=miss.values, y=miss.index, orientation="h",
            marker_color=colors, marker_line_color="white", marker_line_width=1,
            text=[f"{v:.2f}%" for v in miss.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Missing: <b>%{x:.2f}%</b><extra></extra>",
        ))
        fig.add_vline(x=5, line_dash="dot", line_color=TEAL, annotation_text="5%")
        fig.add_vline(x=40, line_dash="dot", line_color=RED, annotation_text="40%")
        fig.update_layout(height=280, margin=dict(t=20,b=20,l=10,r=80),
                          showlegend=False, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                          xaxis=dict(showgrid=True, gridcolor="#EEE", color="#0A1628"),
                          yaxis=dict(color="#0A1628"),
                          font=dict(color="#0A1628"),
                          hoverlabel=dict(bgcolor=NAVY, font_color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Class Imbalance")
        counts = df["default_flag"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=["No Default", "Defaulted"],
            values=counts.values, hole=0.55,
            marker_colors=[GREEN, RED],
            marker_line=dict(color="white", width=3),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig2.update_layout(
            height=280, margin=dict(t=20,b=20,l=10,r=10),
            annotations=[dict(text=f"<b>{df['default_flag'].mean()*100:.1f}%<br>Default</b>",
                              x=0.5, y=0.5, font_size=14, font_color=RED, showarrow=False)],
            paper_bgcolor="white",
            font=dict(color="#0A1628"),
            hoverlabel=dict(bgcolor=NAVY, font_color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(color="#0A1628")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Column Summary")
    summary = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
        "Unique Values": df.nunique(),
        "Mean": df.mean(numeric_only=True).round(2),
    }).sort_values("Missing %", ascending=False)
    st.dataframe(summary, use_container_width=True, height=350)

    st.subheader("Outlier Detection (IQR Method)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["default_flag", "branch_code_id"]]
    outlier_rows = []
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_rows.append({"Feature": col, "Q1": round(Q1,2), "Q3": round(Q3,2),
                              "IQR": round(IQR,2), "Outliers": n_out,
                              "Outlier %": round(n_out/len(df)*100,2)})
    st.dataframe(pd.DataFrame(outlier_rows).sort_values("Outliers", ascending=False),
                 use_container_width=True, height=300)


# ── PAGE: EDA EXPLORER ─────────────────────────────────────────────────────────
elif page == "EDA Explorer":
    st.markdown("""<div class="chapter-header"><h1>EDA Explorer</h1>
    <p>Univariate analysis &nbsp;|&nbsp; Bivariate analysis &nbsp;|&nbsp; WoE/IV rankings</p></div>""",
    unsafe_allow_html=True)

    df, df_train, df_test = get_data()
    md = get_model()
    df_tr = df_train.copy()
    overall_dr = df_tr["default_flag"].mean() * 100

    tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate", "WoE / IV Rankings"])

    with tab1:
        num_cols = [c for c in df_tr.select_dtypes(include=[np.number]).columns
                    if c not in ["default_flag", "branch_code_id"]]
        cat_cols = [c for c in df_tr.select_dtypes(include=["object", "bool"]).columns
                    if c not in ["applicant_id_hash", "application_date", "set"]]

        feat_type = st.radio("Feature type", ["Numeric", "Categorical"], horizontal=True)

        if feat_type == "Numeric":
            selected  = st.selectbox("Select feature", num_cols)
            defaulted = df_tr[df_tr["default_flag"] == 1][selected].dropna()
            non_def   = df_tr[df_tr["default_flag"] == 0][selected].dropna()

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=non_def, name="No Default", marker_color=GREEN,
                    opacity=0.6, histnorm="probability density",
                    hovertemplate=f"<b>No Default</b><br>{selected}: %{{x}}<br>Density: %{{y:.4f}}<extra></extra>"))
                fig.add_trace(go.Histogram(x=defaulted, name="Defaulted", marker_color=RED,
                    opacity=0.6, histnorm="probability density",
                    hovertemplate=f"<b>Defaulted</b><br>{selected}: %{{x}}<br>Density: %{{y:.4f}}<extra></extra>"))
                fig.update_layout(barmode="overlay", title=f"Distribution: {selected}",
                    height=350, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                    margin=dict(t=40,b=20),
                    font=dict(color="#0A1628"),
                    legend=dict(orientation="h",y=-0.2,font=dict(color="#0A1628")),
                    hoverlabel=dict(bgcolor=NAVY, font_color="white"))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                try:
                    _, iv_val  = calculate_iv(df_tr, selected, "default_flag", bins=6, is_categorical=False)
                    woe_grp, _ = calculate_iv(df_tr, selected, "default_flag", bins=6, is_categorical=False)
                    bin_col  = woe_grp.columns[0]
                    labels   = [str(b)[:22] for b in woe_grp[bin_col]]
                    woe_vals = woe_grp["WoE"].values
                    dr_vals  = woe_grp["default_rate"].values * 100
                    counts   = woe_grp["total"].values
                    colors   = [RED if w < 0 else GREEN for w in woe_vals]
                    fig2 = go.Figure(go.Bar(
                        x=woe_vals, y=labels, orientation="h",
                        marker_color=colors, marker_line_color="white", marker_line_width=0.5,
                        text=[f"{w:.3f}" for w in woe_vals], textposition="outside",
                        customdata=list(zip(dr_vals, counts)),
                        hovertemplate="<b>Bin: %{y}</b><br>WoE: <b>%{x:.4f}</b><br>Default Rate: %{customdata[0]:.1f}%<br>Count: %{customdata[1]:,}<extra></extra>",
                    ))
                    fig2.add_vline(x=0, line_color=NAVY, line_width=2)
                    strength = "Strong" if iv_val > 0.30 else "Medium" if iv_val > 0.10 else "Weak" if iv_val > 0.02 else "Useless"
                    fig2.update_layout(title=f"WoE per Bin - IV = {iv_val:.4f} ({strength})",
                        height=350, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                        margin=dict(t=40,b=20,r=80),
                        font=dict(color="#0A1628"),
                        showlegend=False,
                        hoverlabel=dict(bgcolor=NAVY, font_color="white"))
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"WoE chart unavailable: {e}")

            col_stats = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max"],
                "No Default": [f"{non_def.mean():.3f}", f"{non_def.median():.3f}",
                               f"{non_def.std():.3f}", f"{non_def.min():.3f}", f"{non_def.max():.3f}"],
                "Defaulted":  [f"{defaulted.mean():.3f}", f"{defaulted.median():.3f}",
                               f"{defaulted.std():.3f}", f"{defaulted.min():.3f}", f"{defaulted.max():.3f}"],
            })
            st.dataframe(col_stats, use_container_width=True, hide_index=True)

        else:
            selected   = st.selectbox("Select feature", cat_cols)
            dr         = df_tr.groupby(selected)["default_flag"].mean().sort_values(ascending=True) * 100
            cnts       = df_tr.groupby(selected).size()
            bar_colors = [RED if v > overall_dr+1 else GREEN if v < overall_dr-1 else GOLD for v in dr.values]
            fig = go.Figure(go.Bar(
                x=dr.values, y=dr.index, orientation="h",
                marker_color=bar_colors, marker_line_color="white",
                text=[f"{v:.1f}%" for v in dr.values], textposition="outside",
                customdata=[[cnts.get(c,0)] for c in dr.index],
                hovertemplate="<b>%{y}</b><br>Default Rate: <b>%{x:.2f}%</b><br>Count: %{customdata[0]:,}<extra></extra>",
            ))
            fig.add_vline(x=overall_dr, line_dash="dash", line_color=NAVY, line_width=1.5,
                          annotation_text=f"Avg {overall_dr:.1f}%")
            fig.update_layout(title=f"Default Rate by {selected}",
                height=max(300, len(dr)*48), paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                margin=dict(t=40,b=20,r=80),
                font=dict(color="#0A1628"),
                showlegend=False,
                hoverlabel=dict(bgcolor=NAVY, font_color="white"))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols2 = [c for c in df_tr.select_dtypes(include=[np.number]).columns
                     if c not in ["default_flag", "branch_code_id"]]
        c1, c2 = st.columns(2)
        with c1: feat_x = st.selectbox("Feature X", num_cols2, index=0)
        with c2: feat_y = st.selectbox("Feature Y", num_cols2, index=3)

        col_a, col_b = st.columns(2)
        with col_a:
            sample = df_tr.sample(min(3000, len(df_tr)), random_state=42)
            fig = px.scatter(sample, x=feat_x, y=feat_y, color="default_flag",
                color_continuous_scale=[GREEN, RED],
                labels={"default_flag": "Default", feat_x: feat_x, feat_y: feat_y},
                opacity=0.45, title=f"{feat_x} vs {feat_y}")
            fig.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                              margin=dict(t=40,b=20),
                              font=dict(color="#0A1628"),
                              hoverlabel=dict(bgcolor=NAVY, font_color="white"))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            num_for_corr = [c for c in num_cols2 if c in df_tr.columns][:14]
            corr = df_tr[num_for_corr + ["default_flag"]].corr().round(3)
            hover_text = [[
                f"<b>{corr.index[i]}</b> vs <b>{corr.columns[j]}</b><br>r = {corr.iloc[i,j]:.3f}"
                for j in range(len(corr.columns))] for i in range(len(corr.index))]
            fig2 = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                text=hover_text, hoverinfo="text",
                colorscale=[[0, RED],[0.5,"white"],[1, TEAL]],
                zmid=0, zmin=-0.6, zmax=0.6, xgap=1, ygap=1,
                colorbar=dict(title="r", thickness=12),
            ))
            fig2.update_layout(title="Correlation Matrix", height=380,
                paper_bgcolor="white", plot_bgcolor="#FAFAFA", margin=dict(t=40,b=20),
                font=dict(color="#0A1628"),
                xaxis=dict(tickangle=45, showgrid=False, tickfont=dict(size=9, color="#0A1628")),
                yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=9, color="#0A1628")),
                hoverlabel=dict(bgcolor=NAVY, font_color="white"))
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        iv_df    = md["iv_df"]
        iv_sorted= iv_df.sort_values("IV", ascending=True)
        color_map= {"Very Strong":"#8B0000","Strong":RED,"Medium":GOLD,"Weak":TEAL,"Useless":MUTED}
        bar_colors = [color_map.get(s, MUTED) for s in iv_sorted["Strength"]]
        fig = go.Figure(go.Bar(
            x=iv_sorted["IV"], y=iv_sorted["Feature"], orientation="h",
            marker_color=bar_colors, marker_line_color="white",
            text=[f"{v:.4f}  [{s}]" for v, s in zip(iv_sorted["IV"], iv_sorted["Strength"])],
            textposition="outside",
            customdata=iv_sorted[["Strength"]].values,
            hovertemplate="<b>%{y}</b><br>IV: <b>%{x:.4f}</b><br>Strength: %{customdata[0]}<extra></extra>",
        ))
        for xv, lbl, clr in [(0.02,"Weak",MUTED),(0.10,"Medium",TEAL),(0.30,"Strong",GOLD)]:
            fig.add_vline(x=xv, line_dash="dash", line_color=clr,
                          annotation_text=lbl, annotation_position="top right")
        fig.update_layout(title="All Features Ranked by Information Value (IV)",
            height=max(500, len(iv_sorted)*28), paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            margin=dict(t=50,b=20,r=160),
            font=dict(color="#0A1628"),
            showlegend=False,
            hoverlabel=dict(bgcolor=NAVY, font_color="white"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(iv_df, use_container_width=True, hide_index=True)


# ── PAGE: MODEL & SCORECARD ────────────────────────────────────────────────────
elif page == "Model & Scorecard":
    st.markdown("""<div class="chapter-header"><h1>Model & Scorecard</h1>
    <p>Credit scorecard points table &nbsp;|&nbsp; Model performance &nbsp;|&nbsp; LightGBM reference comparison</p></div>""",
    unsafe_allow_html=True)

    md = get_model()

    tab1, tab2, tab3 = st.tabs(["Credit Scorecard", "Model Performance", "LightGBM Comparison"])

    with tab1:
        st.markdown("""
        <div style="background:#F8FAFB; border-left:4px solid #007B8A; padding:16px 20px; border-radius:0 8px 8px 0; margin-bottom:20px;">
            <div style="font-size:14px; font-weight:600; color:#0A1628; margin-bottom:4px;">What is a Credit Scorecard?</div>
            <div style="font-size:13px; color:#444444; line-height:1.7;">
            A credit scorecard converts the logistic regression model into a points-based system.
            Every feature bin is assigned a number of points. The points are added together to produce
            a final credit score between 300 and 900. Higher score = lower risk = more likely to be approved.
            This is the same approach used by real banks — every decision is fully explainable.
            </div>
        </div>
        """, unsafe_allow_html=True)

        lr_model   = md["pipe"].named_steps["lr"]
        feat_names = md["feat_names"]
        coefs      = lr_model.coef_[0]
        intercept  = lr_model.intercept_[0]

        factor    = 20 / np.log(2)
        offset    = 600 - factor * intercept
        coef_df   = md["coef_df"].copy()

        coef_df["Points Contribution"] = (-coef_df["Coefficient"] * factor).round(0).astype(int)
        coef_df["Direction"] = coef_df["Coefficient"].apply(
            lambda x: "Lowers Risk" if x > 0 else "Raises Risk"
        )

        display_df = coef_df[["Feature","Coefficient","Points Contribution","Direction"]].copy()
        display_df.columns = ["Feature (WoE)", "Beta Coefficient", "Score Points", "Risk Direction"]
        display_df = display_df.sort_values("Score Points", ascending=False).reset_index(drop=True)

        st.markdown(f"**Score Range: 300 (very high risk) to 900 (very low risk) | Approval cut-off: ~580**")
        st.markdown(f"**Model Intercept (base score offset): {offset:.0f} points**")
        st.markdown("<br>", unsafe_allow_html=True)

        fig_sc = go.Figure(go.Bar(
            x=display_df["Score Points"],
            y=display_df["Feature (WoE)"],
            orientation="h",
            marker_color=[GREEN if v > 0 else RED for v in display_df["Score Points"]],
            marker_line_color="white", marker_line_width=0.5,
            text=[f"{v:+d} pts" for v in display_df["Score Points"]],
            textposition="outside",
            customdata=display_df[["Beta Coefficient", "Risk Direction"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Score Points: <b>%{x:+d}</b><br>"
                "Beta: %{customdata[0]:.4f}<br>"
                "Direction: %{customdata[1]}"
                "<extra></extra>"
            ),
        ))
        fig_sc.add_vline(x=0, line_color=NAVY, line_width=2)
        fig_sc.update_layout(
            title="Scorecard Points by Feature (Green = Helps Approval, Red = Hurts Approval)",
            height=max(450, len(display_df)*32),
            paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            margin=dict(t=50,b=20,r=100),
            font=dict(color="#0A1628"),
            showlegend=False,
            hoverlabel=dict(bgcolor=NAVY, font_color="white"),
            xaxis_title="Score Points Contribution",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("#### Full Scorecard Table")
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

        st.markdown("#### Score Interpretation Guide")
        c1, c2, c3, c4 = st.columns(4)
        for col, rng, label, clr, meaning in [
            (c1, "300 - 449", "Very High Risk",  RED,   "Almost certainly declined"),
            (c2, "450 - 579", "High Risk",        GOLD,  "Likely declined"),
            (c3, "580 - 649", "Medium Risk",      TEAL,  "Borderline - may be approved"),
            (c4, "650 - 900", "Low Risk",         GREEN, "Strong approval candidate"),
        ]:
            col.markdown(
                f"""<div class="metric-card" style="border-top:4px solid {clr};">
                <div class="metric-label">{rng}</div>
                <div class="metric-value" style="font-size:16px; color:{clr};">{label}</div>
                <div class="metric-sub">{meaning}</div></div>""",
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown("""
        <div style="background:#F8FAFB; border-left:4px solid #007B8A; padding:16px 20px; border-radius:0 8px 8px 0; margin-bottom:20px;">
            <div style="font-size:14px; font-weight:600; color:#0A1628; margin-bottom:4px;">How to Read These Charts</div>
            <div style="font-size:13px; color:#444444; line-height:1.7;">
            These charts show how well the model separates defaulters from non-defaulters.
            The further the curves are from the diagonal line, the better the model is performing.
            All metrics were calculated on the test set — data the model never saw during training.
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, sub, clr in [
            (c1, "AUC",       f"{md['auc']:.4f}",  "0.50 = random, 1.0 = perfect", TEAL),
            (c2, "Gini",      f"{md['gini']:.4f}", "= 2 x AUC - 1",               GOLD),
            (c3, "KS Stat",   f"{md['ks']:.4f}",   "Max good vs bad separation",  GREEN),
            (c4, "Improved",  f"+{md['auc']-0.68:.4f}", "vs baseline AUC of 0.68", NAVY),
        ]:
            col.markdown(
                f"""<div class="metric-card" style="border-top:4px solid {clr};">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{clr};">{val}</div>
                <div class="metric-sub">{sub}</div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("ROC Curve")
            fpr, tpr = md["fpr"], md["tpr"]
            step = max(1, len(fpr)//500)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr[::step], y=tpr[::step], mode="lines",
                name=f"Our Model (AUC = {md['auc']:.4f})",
                line=dict(color=TEAL, width=3),
                fill="tozeroy", fillcolor="rgba(0,123,138,0.08)",
                hovertemplate="False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra>Our Model</extra>",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines", name="Random Model (AUC = 0.50)",
                line=dict(color=MUTED, width=1.5, dash="dot"), hoverinfo="skip",
            ))
            fig_roc.update_layout(
                height=380, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                margin=dict(t=20,b=40,l=40,r=20),
                font=dict(color="#0A1628"),
                xaxis_title="False Positive Rate (% of good customers wrongly declined)",
                yaxis_title="True Positive Rate (% of defaulters caught)",
                legend=dict(orientation="h", y=-0.2, font=dict(color="#0A1628")),
                hoverlabel=dict(bgcolor=NAVY, font_color="white"),
                xaxis=dict(range=[0,1], showgrid=True, gridcolor="#EEE", color="#0A1628"),
                yaxis=dict(range=[0,1], showgrid=True, gridcolor="#EEE", color="#0A1628"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("The higher the curve bows toward the top-left corner, the better the model. The diagonal line represents a model with no skill — just random guessing.")

        with col_b:
            st.subheader("Score Distribution by Default Status")
            proba_te = md["proba_te"]
            y_te     = md["y_te"]
            scores   = (300 + (1 - proba_te) * 600).astype(int)
            defaulters    = scores[y_te == 1]
            non_defaulters= scores[y_te == 0]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=non_defaulters, name="No Default (Good Customers)",
                marker_color=GREEN, opacity=0.6, nbinsx=40,
                hovertemplate="Score: %{x}<br>Count: %{y:,}<extra>Good Customers</extra>",
            ))
            fig_dist.add_trace(go.Histogram(
                x=defaulters, name="Defaulted (Bad Customers)",
                marker_color=RED, opacity=0.6, nbinsx=40,
                hovertemplate="Score: %{x}<br>Count: %{y:,}<extra>Defaulters</extra>",
            ))
            fig_dist.add_vline(x=580, line_dash="dash", line_color=GOLD, line_width=2,
                               annotation_text="Approval cut-off: 580",
                               annotation_font_color=GOLD)
            fig_dist.update_layout(
                barmode="overlay", height=380,
                paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                margin=dict(t=20,b=40,l=40,r=20),
                font=dict(color="#0A1628"),
                xaxis_title="Credit Score",
                yaxis_title="Number of Applicants",
                legend=dict(orientation="h", y=-0.2, font=dict(color="#0A1628")),
                hoverlabel=dict(bgcolor=NAVY, font_color="white"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("A good model pushes green (good customers) to the right and red (defaulters) to the left. The gold dashed line is the approval cut-off — applicants to the right get approved.")

        st.subheader("Confusion Matrix — At 50% Threshold")
        from sklearn.metrics import confusion_matrix
        y_pred = (proba_te >= 0.50).astype(int)
        cm = confusion_matrix(y_te, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        hover_cm = [
            [f"<b>Actual: {['No Default','Defaulted'][i]}</b><br>Predicted: {['No Default','Defaulted'][j]}<br>"
             f"Count: <b>{cm[i,j]:,}</b><br>Row %: <b>{cm_pct[i,j]:.1f}%</b>"
             for j in range(2)] for i in range(2)
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_pct, x=["Predicted: No Default","Predicted: Default"],
            y=["Actual: No Default","Actual: Default"],
            text=hover_cm, hoverinfo="text",
            colorscale=[[0,"#EAF4FB"],[1,TEAL]],
            showscale=False, xgap=2, ygap=2,
        ))
        for i in range(2):
            for j in range(2):
                fig_cm.add_annotation(
                    x=["Predicted: No Default","Predicted: Default"][j],
                    y=["Actual: No Default","Actual: Default"][i],
                    text=f"<b>{cm_pct[i,j]:.1f}%</b><br><span style='font-size:11px'>n={cm[i,j]:,}</span>",
                    showarrow=False,
                    font=dict(size=14, color="white" if cm_pct[i,j] > 50 else NAVY),
                )
        fig_cm.update_layout(
            height=320, paper_bgcolor="white",
            margin=dict(t=20,b=20,l=10,r=10),
            font=dict(color="#0A1628"),
            hoverlabel=dict(bgcolor=NAVY, font_color="white"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Top-left = correct approvals. Bottom-right = correctly caught defaulters. Top-right = good customers wrongly declined. Bottom-left = defaulters wrongly approved.")

    with tab3:
        st.markdown("""
        <div style="background:#FFF3DC; border-left:4px solid #C9A84C; padding:16px 20px; border-radius:0 8px 8px 0; margin-bottom:20px;">
            <div style="font-size:14px; font-weight:600; color:#7B4800; margin-bottom:4px;">Why We Show LightGBM</div>
            <div style="font-size:13px; color:#444444; line-height:1.7;">
            LightGBM is a powerful machine learning model with no interpretability constraints.
            We trained it purely as a performance ceiling — to show how close our logistic regression
            gets to the best possible result. We are NOT submitting LightGBM. Our final model is
            logistic regression because regulators and risk managers require full explainability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        baseline_auc = 0.68
        our_auc      = md["auc"]
        lgb_auc      = 0.82
        gap_closed   = (our_auc - baseline_auc) / (lgb_auc - baseline_auc) * 100

        models = ["Baseline LR (Given benchmark)", f"Our Improved LR (This project)", "LightGBM (Ceiling - not submitted)"]
        aucs   = [baseline_auc, our_auc, lgb_auc]
        ginis  = [round(2*a-1, 4) for a in aucs]
        colors = [MUTED, TEAL, GOLD]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="AUC", x=models, y=aucs,
            marker_color=colors, marker_line_color="white", marker_line_width=1,
            text=[f"AUC: {a:.4f}<br>Gini: {g:.4f}" for a, g in zip(aucs, ginis)],
            textposition="outside",
            customdata=list(zip(aucs, ginis)),
            hovertemplate="<b>%{x}</b><br>AUC: <b>%{customdata[0]:.4f}</b><br>Gini: <b>%{customdata[1]:.4f}</b><extra></extra>",
            width=0.45,
        ))
        fig_comp.add_hline(y=0.68, line_dash="dot", line_color=MUTED, line_width=1.5,
                           annotation_text="Baseline: 0.68")
        fig_comp.add_hline(y=0.82, line_dash="dot", line_color=GOLD, line_width=1.5,
                           annotation_text="LightGBM ceiling: 0.82")
        fig_comp.update_layout(
            title="Model Performance Comparison — Baseline vs Our LR vs LightGBM Ceiling",
            height=420, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            margin=dict(t=60,b=40,l=40,r=40),
            font=dict(color="#0A1628"),
            yaxis=dict(range=[0.60, 0.88], title="AUC Score", showgrid=True, gridcolor="#EEE", color="#0A1628"),
            xaxis=dict(showgrid=False, color="#0A1628"),
            showlegend=False,
            hoverlabel=dict(bgcolor=NAVY, font_color="white"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown(f"#### Gap Closed: {gap_closed:.1f}% of the distance from baseline to LightGBM ceiling")
        st.progress(int(gap_closed))
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:16px;">
            <div class="metric-card" style="border-top:4px solid {MUTED};">
                <div class="metric-label">Baseline LR AUC</div>
                <div class="metric-value" style="color:{MUTED};">0.6800</div>
                <div class="metric-sub">The starting point given by the competition</div>
            </div>
            <div class="metric-card" style="border-top:4px solid {TEAL};">
                <div class="metric-label">Our Improved LR AUC</div>
                <div class="metric-value" style="color:{TEAL};">{our_auc:.4f}</div>
                <div class="metric-sub">Achieved through WoE feature engineering</div>
            </div>
            <div class="metric-card" style="border-top:4px solid {GOLD};">
                <div class="metric-label">LightGBM Ceiling AUC</div>
                <div class="metric-value" style="color:{GOLD};">0.8200</div>
                <div class="metric-sub">Best possible — unconstrained ML model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        #### Why Not Just Use LightGBM?

        | Factor | Our Logistic Regression | LightGBM |
        |--------|------------------------|----------|
        | Can explain every decision? | Yes — every coefficient has meaning | No — black box |
        | Regulatory compliant (Basel III/IV)? | Yes | No |
        | Auditable by NCR/SARB? | Yes | No |
        | Customer can be told why they were declined? | Yes | No |
        | Performance (AUC)? | Strong | Slightly better |
        | **Overall winner for credit lending?** | **Yes** | **No** |

        The table above shows why logistic regression wins in regulated credit environments despite
        slightly lower raw performance. A model that cannot explain itself cannot be deployed in a bank.
        """)


# ── PAGE: BUSINESS DASHBOARD ───────────────────────────────────────────────────
elif page == "Business Dashboard":
    st.markdown("""<div class="chapter-header"><h1>Business Decision Dashboard</h1>
    <p>Threshold analysis &nbsp;|&nbsp; Rand-value impact &nbsp;|&nbsp; Volume vs Risk</p></div>""",
    unsafe_allow_html=True)

    md = get_model()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "AUC",      f"{md['auc']:.4f}",  "Area Under Curve"),
        (c2, "Gini",     f"{md['gini']:.4f}", "2 x AUC minus 1"),
        (c3, "KS",       f"{md['ks']:.4f}",   "Max separation"),
        (c4, "Baseline", "0.6800",             "Competition benchmark"),
    ]:
        col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_roc, col_thresh = st.columns([3, 2])
    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr = md["fpr"], md["tpr"]
        step = max(1, len(fpr)//500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr[::step], y=tpr[::step], mode="lines",
            name=f"Improved LR (AUC={md['auc']:.4f})", line=dict(color=TEAL, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>Improved LR</extra>"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"))
        fig.update_layout(height=340, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            margin=dict(t=20,b=20),
            font=dict(color="#0A1628"),
            xaxis_title="FPR", yaxis_title="TPR",
            legend=dict(orientation="h", y=-0.2, font=dict(color="#0A1628")),
            hoverlabel=dict(bgcolor=NAVY, font_color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with col_thresh:
        st.subheader("Set Approval Threshold")
        threshold = st.slider("Threshold", 0.20, 0.80, 0.50, 0.01,
            help="Move left = approve more. Move right = approve fewer.")
        AVG_LOAN = st.number_input("Avg loan size (R)", 5000, 100000, 15000, 1000)
        INTEREST = st.number_input("Annual interest rate (%)", 5.0, 30.0, 15.0, 0.5)
        LGD      = st.number_input("Loss Given Default (%)", 20, 100, 60, 5) / 100

    proba = md["proba_te"]
    y_te  = md["y_te"]
    thresholds = np.arange(0.20, 0.81, 0.05)
    results = []
    for t in thresholds:
        y_pred   = (proba >= t).astype(int)
        approved = (y_pred == 0).sum()
        declined = (y_pred == 1).sum()
        mask     = y_pred == 0
        true_def = y_te[mask].sum()
        revenue  = (mask.sum() - true_def) * AVG_LOAN * (INTEREST/100)
        bad_debt = true_def * AVG_LOAN * LGD
        results.append({"Threshold": round(float(t),2), "Approved": int(approved),
            "Declined": int(declined), "Approval Rate (%)": round(approved/len(y_pred)*100,1),
            "Expected Defaults": int(true_def), "Revenue (R)": revenue,
            "Bad Debt (R)": bad_debt, "Net (R)": revenue - bad_debt})
    biz_df = pd.DataFrame(results)

    y_pred_now = (proba >= threshold).astype(int)
    approved_n = (y_pred_now == 0).sum()
    declined_n = (y_pred_now == 1).sum()
    mask_n     = y_pred_now == 0
    true_def_n = y_te[mask_n].sum()
    revenue_n  = (mask_n.sum() - true_def_n) * AVG_LOAN * (INTEREST/100)
    bad_debt_n = true_def_n * AVG_LOAN * LGD
    net_n      = revenue_n - bad_debt_n

    st.markdown("#### Current Threshold Impact")
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, clr in [
        (c1, "Approved",     f"{approved_n:,}",        GREEN),
        (c2, "Declined",     f"{declined_n:,}",        RED),
        (c3, "Exp. Defaults",f"{true_def_n:,}",        RED),
        (c4, "Revenue",      f"R{revenue_n/1e6:.2f}M", GREEN),
        (c5, "Net Position", f"R{net_n/1e6:.2f}M",    GREEN if net_n>0 else RED),
    ]:
        col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
        <div class="metric-value" style="font-size:20px; color:{clr};">{val}</div></div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=biz_df["Threshold"], y=biz_df["Revenue (R)"]/1e6,
            mode="lines+markers", name="Revenue (R Millions)", line=dict(color=GREEN,width=2.5),
            marker=dict(size=8), hovertemplate="Threshold: %{x:.2f}<br>Revenue: R%{y:.3f}M<extra></extra>"))
        fig.add_trace(go.Scatter(x=biz_df["Threshold"], y=biz_df["Bad Debt (R)"]/1e6,
            mode="lines+markers", name="Bad Debt (R Millions)", line=dict(color=RED,width=2.5,dash="dash"),
            marker=dict(size=8), hovertemplate="Threshold: %{x:.2f}<br>Bad Debt: R%{y:.3f}M<extra></extra>"))
        fig.add_vline(x=threshold, line_dash="dot", line_color=GOLD, line_width=2,
                      annotation_text=f"Current: {threshold}")
        fig.update_layout(title="Revenue vs Bad Debt by Threshold", height=320,
            paper_bgcolor="white", plot_bgcolor="#FAFAFA", margin=dict(t=40,b=20),
            font=dict(color="#0A1628"),
            legend=dict(orientation="h",y=-0.25,font=dict(color="#0A1628")),
            hoverlabel=dict(bgcolor=NAVY, font_color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        net_vals = biz_df["Net (R)"] / 1e6
        bar_clrs = [GREEN if v >= 0 else RED for v in net_vals]
        fig2 = go.Figure(go.Bar(
            x=biz_df["Threshold"].astype(str), y=net_vals,
            marker_color=bar_clrs, marker_line_color="white",
            text=[f"R{v:.2f}M" for v in net_vals], textposition="outside",
            hovertemplate="Threshold: %{x}<br>Net: R%{y:.3f}M<extra></extra>"))
        fig2.add_hline(y=0, line_color=NAVY, line_width=1.5)
        fig2.update_layout(title="Net Financial Position by Threshold", height=320,
            paper_bgcolor="white", plot_bgcolor="#FAFAFA", margin=dict(t=40,b=20),
            font=dict(color="#0A1628"),
            showlegend=False,
            hoverlabel=dict(bgcolor=NAVY, font_color="white"))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Precision / Recall - Business Meaning")
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_te, y_pred_now, zero_division=0)
    rec  = recall_score(y_te, y_pred_now, zero_division=0)
    f1   = f1_score(y_te, y_pred_now, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card"><div class="metric-label">Precision</div>
    <div class="metric-value">{prec:.3f}</div>
    <div class="metric-sub">Of declined applicants, {prec*100:.1f}% would actually default.</div></div>""",
    unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><div class="metric-label">Recall</div>
    <div class="metric-value">{rec:.3f}</div>
    <div class="metric-sub">Of all actual defaulters, {rec*100:.1f}% are correctly caught.</div></div>""",
    unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card"><div class="metric-label">F1 Score</div>
    <div class="metric-value">{f1:.3f}</div>
    <div class="metric-sub">Balance between precision and recall at this threshold.</div></div>""",
    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(biz_df.style.format({
        "Revenue (R)":"R{:,.0f}","Bad Debt (R)":"R{:,.0f}","Net (R)":"R{:,.0f}",
        "Approval Rate (%)":"{:.1f}%"}), use_container_width=True, hide_index=True)


# ── PAGE: CREDIT AI CHATBOT ────────────────────────────────────────────────────
elif page == "Credit AI Chatbot":
    st.markdown("""<div class="chapter-header"><h1>Credit AI Chatbot</h1>
    <p>Ask anything about loans, credit scores, or this model in plain language</p></div>""",
    unsafe_allow_html=True)

    md_model = get_model()
    groq_key = _get_groq_key()

    if not groq_key:
        st.error(
            "⚠️  Groq API key not found. "
            "Add `GROQ_API_KEY = 'your_key'` to `.streamlit/secrets.toml` locally, "
            "or paste it into **Settings → Secrets** on Streamlit Cloud.",
            icon="🔑",
        )

    coef_context = md_model["coef_df"].head(12).to_string(index=False)
    iv_context   = md_model["iv_df"].head(10).to_string(index=False)

    system_prompt = f"""You are a friendly credit risk assistant helping people understand loans and credit scoring.
You work for the FNB DataQuest 2026 project built by Moloi Qolani Truelove and Tshegofatso Tshepang Chikwane from Sol Plaatje University.

MODEL FACTS:
- Model type: Logistic Regression with WoE features
- AUC: {md_model['auc']:.4f} (higher is better, max is 1.0)
- Gini: {md_model['gini']:.4f}
- KS Statistic: {md_model['ks']:.4f}
- Baseline AUC was 0.68, we improved it significantly
- LightGBM ceiling is 0.82

TOP PREDICTORS BY IMPORTANCE:
{iv_context}

TOP MODEL COEFFICIENTS:
{coef_context}

DATASET FACTS:
- 120,960 loan applications
- 15.4% of people defaulted (did not pay back)
- We predict if someone will default within 12 months

BUSINESS FACTS:
- Average loan size: R15,000
- Interest rate: 15% per year
- If someone defaults, the bank loses about 60% of the loan amount

YOUR PERSONALITY AND RULES:
- Always explain things in very simple, everyday language
- Avoid technical jargon unless the person asks for it
- Use short sentences and real examples with Rand amounts
- Be warm, helpful, and encouraging
- If someone asks what a term means, explain it like you are talking to someone with no finance background
- Never make up numbers - only use the facts above
- Keep answers short and to the point (3-5 sentences max unless they ask for more detail)"""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("**Not sure what to ask? Try one of these:**")

    suggestions = [
        "Why was I rejected for a loan?",
        "What is a credit score?",
        "How can I improve my chances of getting a loan?",
        "What happens if I don't pay back my loan?",
        "What does DTI ratio mean?",
        "How does the model decide who gets approved?",
        "What is a good credit score?",
        "Why does my income matter for a loan?",
        "What is a default?",
        "How many people in this dataset did not pay back?",
        "What is the most important thing for loan approval?",
        "How does missing a payment affect me?",
    ]

    rows = [suggestions[i:i+3] for i in range(0, len(suggestions), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, suggestion in zip(cols, row):
            with col:
                if st.button(suggestion, use_container_width=True):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": suggestion}
                    )
                    st.session_state["pending_response"] = True
                    st.rerun()

    st.markdown("---")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user"><b>You:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bot"><b>Assistant:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    if st.session_state.get("pending_response") and st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        if last_msg["role"] == "user":
            if not groq_key:
                response_text = "⚠️ No API key configured. Please add GROQ_API_KEY to your Streamlit secrets."
            else:
                try:
                    from groq import Groq
                    client  = Groq(api_key=groq_key)
                    messages = [{"role": "system", "content": system_prompt}]
                    messages += [{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat_history]
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        max_tokens=600,
                    )
                    response_text = resp.choices[0].message.content
                except Exception as e:
                    response_text = f"Sorry, I could not connect right now. Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.session_state["pending_response"] = False
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your question here",
            placeholder="e.g. Why was my loan rejected?",
        )
        col_send, col_clear = st.columns([4, 1])
        with col_send:
            send = st.form_submit_button("Send", use_container_width=True)
        with col_clear:
            clear = st.form_submit_button("Clear Chat")

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if not groq_key:
            response_text = "⚠️ No API key configured. Please add GROQ_API_KEY to your Streamlit secrets."
        else:
            try:
                from groq import Groq
                client  = Groq(api_key=groq_key)
                messages = [{"role": "system", "content": system_prompt}]
                messages += [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history
                ]
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=600,
                )
                response_text = resp.choices[0].message.content
            except Exception as e:
                response_text = f"Sorry, I could not connect to the AI service right now. Error: {e}"

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text}
        )
        st.rerun()


# ── PAGE: LOAN DECISION REPORT ─────────────────────────────────────────────────
elif page == "Loan Decision Report":
    st.markdown("""<div class="chapter-header"><h1>Loan Decision Report</h1>
    <p>Enter applicant details &nbsp;|&nbsp; Get a credit score &nbsp;|&nbsp; Download a full decision report</p></div>""",
    unsafe_allow_html=True)

    md_model = get_model()

    with st.form("applicant_form"):
        st.markdown("""
        <div style="background:#E8F4F6; border-radius:10px; padding:20px 24px; margin-bottom:20px;">
            <div style="color:#007B8A; font-size:11px; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px;">Step 1 of 2</div>
            <div style="font-size:20px; font-weight:700; color:#0A1628;">Applicant Profile</div>
            <div style="font-size:13px; color:#444444; margin-top:4px;">Fill in all fields below. The model will score this applicant instantly.</div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_name          = st.text_input("Full Name", "", placeholder="e.g. Moloi Qolani Truelove")
            age                     = st.number_input("Age", 18, 80, 32)
            annual_income           = st.number_input("Annual Income (R)", 10000, 2000000, 30000, 1000)
            employment_length_years = st.number_input("Employment Length (years)", 0.0, 50.0, 2.0, 0.5)
        with c2:
            loan_amount         = st.number_input("Loan Amount (R)", 1000, 500000, 10000, 500)
            interest_rate_input = st.number_input("Interest Rate (% p.a.)", 5.0, 40.0, 15.0, 0.5)
            loan_term           = st.selectbox("Loan Term (months)", [6,12,18,24,36,48,60], index=1)
            loan_purpose        = st.selectbox("Loan Purpose", [
                "debt_consolidation","major_purchase","medical",
                "home_improvement","education","small_business","other"])
        with c3:
            dti_ratio              = st.number_input("DTI Ratio (0-1)", 0.0, 2.0, 0.30, 0.01)
            credit_utilisation_pct = st.number_input("Credit Utilisation (%)", 0.0, 100.0, 35.0, 1.0)
            num_delinquencies_2yr  = st.number_input("Delinquencies (last 2 yrs)", 0, 20, 0)
            has_delinquency        = st.checkbox("Has Prior Delinquency?", False)

        st.markdown("""
        <div style="background:#F8FAFB; border-left:4px solid #C9A84C; padding:14px 20px; border-radius:0 8px 8px 0; margin:20px 0 16px;">
            <div style="color:#7B4800; font-size:11px; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">Step 2 of 2</div>
            <div style="font-size:16px; font-weight:700; color:#0A1628;">Additional Profile</div>
        </div>
        """, unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            home_ownership = st.selectbox("Home Ownership", ["MORTGAGE","RENT","OWN","OTHER"])
            region         = st.selectbox("Region", [
                "North-Urban","South-Urban","East-Urban","West-Urban","Central-Urban",
                "North-Suburban","South-Suburban","East-Suburban","West-Suburban","Central-Suburban"])
        with c5:
            num_open_accounts           = st.number_input("Open Accounts", 0, 30, 5)
            months_since_oldest_account = st.number_input("Months Since Oldest Account", 0, 600, 120)
            num_hard_inquiries_6mo      = st.number_input("Hard Inquiries (6 months)", 0, 20, 1)
        with c6:
            email_domain_type    = st.selectbox("Email Domain Type", ["free","corporate","other"])
            phone_verified       = st.checkbox("Phone Verified?", True)
            pct_accounts_current = st.number_input("Accounts Current (%)", 0.0, 100.0, 85.0, 1.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Credit Score Analysis", use_container_width=True)

    if submitted:
        applicant = {
            "age": age, "annual_income": annual_income,
            "employment_length_years": employment_length_years,
            "loan_amount": loan_amount, "interest_rate": interest_rate_input,
            "dti_ratio": dti_ratio, "credit_utilisation_pct": credit_utilisation_pct,
            "num_delinquencies_2yr": num_delinquencies_2yr,
            "has_delinquency": int(has_delinquency),
            "home_ownership": home_ownership, "region": region,
            "loan_purpose": loan_purpose, "num_open_accounts": num_open_accounts,
            "months_since_oldest_account": months_since_oldest_account,
            "num_hard_inquiries_6mo": num_hard_inquiries_6mo,
            "email_domain_type": email_domain_type, "phone_verified": phone_verified,
            "pct_accounts_current": pct_accounts_current,
            "total_revolving_balance": credit_utilisation_pct * annual_income / 100,
        }

        result     = predict_applicant(md_model, applicant)
        prob       = result["probability"]
        score      = result["score"]
        decision   = result["decision"]
        risk_tier  = result["risk_tier"]
        tier_color = result["tier_color"]
        sc_df      = result["scorecard"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0A1628; border-radius:12px; padding:16px 24px; margin-bottom:20px;">
            <div style="color:#C9A84C; font-size:11px; letter-spacing:2px; text-transform:uppercase;">Credit Assessment Result</div>
            <div style="color:#ffffff; font-size:13px; margin-top:4px;">Based on the applicant profile entered above</div>
        </div>
        """, unsafe_allow_html=True)

        col_dec, col_score, col_prob = st.columns([2,1,1])
        with col_dec:
            if decision == "APPROVED":
                st.markdown('<div class="decision-approved">APPROVED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="decision-declined">DECLINED</div>', unsafe_allow_html=True)
        with col_score:
            st.markdown(f"""
            <div class="score-ring" style="background:{tier_color};">
                <div style="color:white;font-size:28px;font-weight:700;">{score}</div>
                <div style="color:rgba(255,255,255,0.8);font-size:9px;letter-spacing:1px;">SCORE</div>
            </div>
            <div style="text-align:center;font-size:13px;font-weight:600;color:{tier_color};">{risk_tier}</div>
            """, unsafe_allow_html=True)
        with col_prob:
            st.metric("Default Probability", f"{prob*100:.1f}%",
                      delta="Threshold: 50.0%", delta_color="inverse")

        st.markdown("#### Score Breakdown - What drove this decision?")
        if not sc_df.empty:
            fig = go.Figure(go.Bar(
                x=sc_df["woe"].head(10), y=sc_df["feature"].head(10), orientation="h",
                marker_color=[RED if w < 0 else GREEN for w in sc_df["woe"].head(10)],
                marker_line_color="white", marker_line_width=0.5,
                text=[f"{w:+.3f}" for w in sc_df["woe"].head(10)], textposition="outside",
                customdata=[[v] for v in sc_df["value"].head(10).astype(str).tolist()],
                hovertemplate="<b>%{y}</b><br>WoE: %{x:.4f}<br>Value: %{customdata[0]}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color=NAVY, line_width=2)
            fig.update_layout(
                title="Top 10 Feature Contributions (Green = Lower Risk, Red = Higher Risk)",
                height=340, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                margin=dict(t=40,b=20,r=80),
                font=dict(color="#0A1628"),
                showlegend=False,
                hoverlabel=dict(bgcolor=NAVY, font_color="white"))
            st.plotly_chart(fig, use_container_width=True)

        top_factors = []
        for _, row in sc_df.head(5).iterrows():
            positive = row["woe"] >= 0
            detail_map = {
                "dti_ratio": f"DTI of {dti_ratio:.2f} {'is within acceptable range' if positive else 'exceeds safe threshold of 0.35'}",
                "credit_utilisation_pct": f"Credit utilisation of {credit_utilisation_pct:.0f}% {'is well managed' if positive else 'is elevated - ideally below 70%'}",
                "age": f"Age {age} {'places you in a lower-risk demographic' if positive else 'is in a higher-risk bracket'}",
                "annual_income": f"Annual income of R{annual_income:,} {'is sufficient relative to debt load' if positive else 'appears low relative to loan obligations'}",
                "employment_length_years": f"Employment of {employment_length_years:.1f} years {'demonstrates stability' if positive else 'is limited - longer tenure lowers risk'}",
                "has_delinquency": f"{'Prior delinquency history increases risk' if not positive else 'No prior delinquency - positive signal'}",
                "num_delinquencies_2yr": f"{num_delinquencies_2yr} delinquencies in 2 years {'is clean' if positive else '- each significantly raises risk'}",
            }
            top_factors.append({
                "name": row["feature"].replace("_", " ").title(),
                "detail": detail_map.get(row["feature"], f"Value: {row['value']} | WoE: {row['woe']:.3f}"),
                "positive": positive,
            })

        st.markdown("---")
        st.markdown("### Download Decision Report")

        if decision == "APPROVED":
            html = generate_approved_report(
                applicant_name=applicant_name, loan_amount=loan_amount,
                interest_rate=interest_rate_input, loan_term_months=loan_term,
                score=score, risk_tier=risk_tier, top_factors=top_factors)
        else:
            improvement_steps = [
                {"icon":"01","title":"Reduce Credit Utilisation",
                 "detail":f"Your current utilisation is {credit_utilisation_pct:.0f}%. Aim to reduce this below 70%, ideally below 50%. Pay down revolving balances before reapplying.",
                 "score_gain":35},
                {"icon":"02","title":"Lower Your Debt-to-Income Ratio",
                 "detail":f"Your DTI of {dti_ratio:.2f} means {dti_ratio*100:.0f}% of your income goes to debt. Pay off existing debts or increase your income before reapplying.",
                 "score_gain":40},
                {"icon":"03","title":"Build a Clean Repayment Record",
                 "detail":"Zero missed payments for 6 months on all accounts. Set up debit orders where possible. Consistent on-time payments rebuild your score fastest.",
                 "score_gain":30},
            ]
            html = generate_declined_report(
                applicant_name=applicant_name, score=score, risk_tier=risk_tier,
                top_negative_factors=[f for f in top_factors if not f.get("positive",True)][:3] or top_factors[:3],
                improvement_steps=improvement_steps)

        b64      = base64.b64encode(html.encode()).decode()
        filename = f"LoanReport_{applicant_name.replace(' ','_')}_{decision}.html"
        bg_color = GREEN if decision == "APPROVED" else RED
        href = (
            '<a href="data:text/html;base64,' + b64 + '" '
            'download="' + filename + '" '
            'style="display:inline-block; background:' + bg_color + '; color:#ffffff; '
            'padding:14px 32px; border-radius:8px; text-decoration:none; '
            'font-weight:600; font-size:15px; letter-spacing:0.5px;">'
            'Download ' + decision + ' Report</a>'
        )
        st.markdown(href, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Report Preview")
        st.components.v1.html(html, height=600, scrolling=True)
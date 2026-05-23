# streamlit_app/app.py
# Flight Delay Predictor — Streamlit Web App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.markdown("""
<style>

/* Main app */
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    color: #0f172a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b, #0f172a);
    color: white;
    border-right: 1px solid #334155;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main titles */
h1 {
    color: #0f172a !important;
    font-size: 52px !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #1e293b !important;
}

/* Cards / metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.75);
    border: 1px solid #cbd5e1;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

/* Input boxes */
.stSelectbox div[data-baseweb="select"],
.stNumberInput input,
.stTextInput input {
    background-color: white !important;
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
}

/* Sliders */
.stSlider > div > div {
    color: #2563eb !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0px 6px 18px rgba(37,99,235,0.3);
}

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg,#2563eb,#06b6d4);
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(37,99,235,0.25);
}

/* Labels */
label, p {
    color: #334155 !important;
    font-weight: 500;
}

/* Horizontal line */
hr {
    border: 1px solid #cbd5e1;
}

</style>
""", unsafe_allow_html=True)



# ── Page configuration ───────────────────────────────────
# This MUST be the first Streamlit command in the file
# layout='wide' uses full screen width instead of narrow column
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈",
    layout="wide"
)


# ── Load the saved model ─────────────────────────────────
# @st.cache_resource means: load this only ONCE when app starts
# Without it, the model reloads every time user clicks anything
# On 50k row model this would make app feel very slow
@st.cache_resource
def load_model():
    return joblib.load('streamlit_app/model.pkl')

bundle     = load_model()
model      = bundle['model']
FEATURES   = bundle['features']
MODEL_NAME = bundle['model_name']
ACCURACY   = bundle['accuracy']
ROC_AUC    = bundle['roc_auc']

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=120)

    st.title("About Project")

    st.write("""
    • ML Model: Random Forest  
    • Accuracy: 83%  
    • ROC-AUC: 89%  
    • Dataset: 50,000+ flights  
    • Tools: Python, SQL, Streamlit, Power BI
    """)

    st.markdown("---")

    st.write("Created by Amisha")

# ── These lists must match EXACTLY what Label Encoder used
# The index position = the encoded number the model learned
# So AIRLINES[0] = 'Alaska' = airline_enc value 0 in training
AIRLINES = ['Alaska','Allegiant','American','Delta',
            'Frontier','Hawaiian','JetBlue','Southwest',
            'Spirit','United']

AIRPORTS  = ['ATL','AUS','BNA','BOS','BWI','CLT','DCA',
             'DEN','DFW','DTW','EWR','FLL','HNL','IAH',
             'JFK','LAS','LAX','MCO','MDW','MIA','MSP',
             'ORD','PDX','PHX','SAN','SEA','SFO','SLC',
             'STL','TPA']

# ── App header ────────────────────────────────────────────
st.title("Flight Delay Predictor")
st.markdown(f"Powered by **{MODEL_NAME}** · Accuracy: **{ACCURACY:.1f}%** · ROC-AUC: **{ROC_AUC:.1f}%**")
# KPI Cards
c1, c2, c3 = st.columns(3)

c1.metric("Model Accuracy", "83%")
c2.metric("ROC-AUC", "89%")
c3.metric("Flights Analyzed", "50,000+")

st.markdown("---")



# ── Input form — 3 columns layout ────────────────────────
st.subheader("Enter flight details")

# st.columns(3) creates 3 equal columns
# col1, col2, col3 are the three column objects
# Anything inside "with col1:" appears in the left column
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Flight Info**")

    # selectbox = dropdown menu
    # First arg = label shown above dropdown
    # Second arg = list of options
    airline  = st.selectbox("Airline", AIRLINES)

    origin   = st.selectbox("Origin Airport", AIRPORTS)

    # Filter out origin from destination options
    # So user cannot pick same airport for both
    dest_options = [a for a in AIRPORTS if a != origin]
    destination  = st.selectbox("Destination Airport", dest_options)

    # number_input = text box where user types a number
    # min_value, max_value = allowed range
    # value = default starting value
    distance = st.number_input(
        "Distance (miles)",
        min_value=100, max_value=3000, value=800, step=50)

with col2:
    st.markdown("**Schedule**")

    # slider = draggable bar
    # Args: label, min, max, default value
    month = st.slider("Month", 1, 12, 6)
    month_names = ['','Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    st.caption(f"Selected: {month_names[month]}")

    dow = st.selectbox(
        "Day of Week",
        options=[1,2,3,4,5,6,7],
        format_func=lambda x: ['','Monday','Tuesday','Wednesday',
                                   'Thursday','Friday','Saturday',
                                   'Sunday'][x]
    )

    hour = st.slider("Departure Hour", 0, 23, 8)
    st.caption(f"Selected: {hour:02d}:00")

    dep_delay = st.slider(
        "Departure Delay (minutes)", -30, 120, 0)
    if dep_delay < 0:
        st.caption(f"Early by {abs(dep_delay)} mins")
    elif dep_delay == 0:
        st.caption("On time departure")
    else:
        st.caption(f"Already delayed by {dep_delay} mins")

# col3 is where the prediction result will appear
# We leave it empty for now — Block 3 fills it


with col3:
    st.markdown("**Prediction**")

    # ── Encode text inputs to numbers ────────────────────
    # AIRLINES.index('Delta') returns 3 — same as label encoder
    # This must match EXACTLY what training used
    airline_enc  = AIRLINES.index(airline)
    origin_enc   = sorted(AIRPORTS).index(origin)
    dest_enc     = sorted(AIRPORTS).index(destination)

    # is_weekend: 1 if Saturday(6) or Sunday(7)
    is_weekend   = 1 if dow >= 6 else 0

    # is_peak_hour: 1 if 6-9am or 4-8pm
    is_peak_hour = 1 if (6<=hour<=9 or 16<=hour<=20) else 0

    # is_long_haul: 1 if over 1500 miles
    is_long_haul = 1 if distance > 1500 else 0

    # season_enc: Fall=0, Spring=1, Summer=2, Winter=3 (alphabetical)
    if   month in [12,1,2]:  season_enc = 3  # Winter
    elif month in [3,4,5]:   season_enc = 1  # Spring
    elif month in [6,7,8]:   season_enc = 2  # Summer
    else:                      season_enc = 0  # Fall

    # tod_enc: Afternoon=0, Evening=1, Morning=2, Night=3 (alphabetical)
    if    0  <= hour < 6:   tod_enc = 3  # Night
    elif  6  <= hour < 12:  tod_enc = 2  # Morning
    elif  12 <= hour < 18:  tod_enc = 0  # Afternoon
    else:                    tod_enc = 1  # Evening

    # ── Build input DataFrame ────────────────────────────
    # Column order MUST match FEATURES list from training exactly
    input_data = pd.DataFrame([[ hour, dow, month, distance,
        airline_enc, origin_enc, dest_enc,
        is_weekend, is_peak_hour, is_long_haul,
        season_enc, tod_enc
    ]], columns=FEATURES)

    # ── Predict ──────────────────────────────────────────
    # predict_proba returns [prob_on_time, prob_delayed]
    # We take index [0][1] = probability of being delayed
    prob = model.predict_proba(input_data)[0][1]
    pct  = prob * 100

    # ── Show result ──────────────────────────────────────
    st.metric(label="Delay Probability", value=f"{pct:.1f}%")

    if pct >= 60:
        st.error(f"HIGH delay risk ({pct:.0f}%)")
        risk = "HIGH"
    elif pct >= 35:
        st.warning(f"MODERATE delay risk ({pct:.0f}%)")
        risk = "MODERATE"
    else:
        st.success(f"LOW delay risk ({pct:.0f}%)")
        risk = "LOW"

    # ── Mini progress bar showing probability ────────────
    st.progress(int(pct))

    # ── Show what drove this prediction ──────────────────
    st.markdown("**Key factors:**")
    if dep_delay > 15:
        st.caption(f"Already {dep_delay} mins late departing")
    if hour >= 18:
        st.caption("Evening flight — higher delay risk")
    if airline in ['Spirit','Frontier','Allegiant']:
        st.caption(f"{airline} has above-average delay rate")
    if month in [12,1,2]:
        st.caption("Winter month — weather delays more likely")
    if dow == 5:
        st.caption("Friday — busiest travel day")

        ''
        ''
        ''
        # ── Historical context section ────────────────────────────
# Load clean data for context charts
# @st.cache_data caches the dataframe so it doesn't reload
# every time the user interacts with a widget
@st.cache_data
def load_data():
    return pd.read_csv('data/flights_clean.csv')

df = load_data()

st.markdown("---")
st.subheader("Historical context")

# Two columns for historical charts
h1, h2 = st.columns(2)

with h1:
    # Bar chart: delay rate for each airline
    st.markdown(f"**Delay rate by airline** (your selection: {airline})")
    airline_rates = (df.groupby('airline')['is_delayed']
                       .mean() * 100
                       ).round(1)
    colors = ['#F44336' if a == airline else '#B5D4F4'
              for a in airline_rates.index]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(airline_rates.index, airline_rates.values, color=colors)
    ax.set_xlabel('Delay Rate (%)')
    ax.set_title('Historical delay rate by airline')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with h2:
    # Line chart: delay rate by hour
    st.markdown(f"**Delay rate by hour** (your selection: {hour:02d}:00)")
    hourly = df.groupby('hour')['is_delayed'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(hourly.index, hourly.values, color='#B5D4F4', lw=2)
    ax.axvline(hour, color='#F44336', ls='--', lw=2,
               label=f'Your hour: {hour:02d}:00')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Delay Rate (%)')
    ax.set_title('Historical delay rate by hour')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
### ✈️ About this project

Analyzed 50,000+ US flights (2022–2024) across 10 airlines and 30 airports.

Built an end-to-end Machine Learning flight delay prediction system using:

✅ Python  
✅ SQL  
✅ Streamlit  
✅ Power BI  
✅ Tableau  
✅ Random Forest Machine Learning Model

The project compares multiple ML algorithms including XGBoost and Random Forest, with Random Forest selected as the final model due to better prediction accuracy and performance.
    """
)
st.markdown("""
---
### 👩‍💻 Developed By
**Amisha **

🔗 [LinkedIn](https://www.linkedin.com/in/amisha-kaliraman-16b30a327)

💻 [GitHub Repository](YOUR_GITHUB_LINK)
""")
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(page_title="EduPro AI", layout="wide")

# ================= YOUR ORIGINAL CSS =================
st.markdown("""
<style>

/* APP BACKGROUND */
.stApp {
    background: #0e1117;
    color: #e6e6e6;
}

/* MAIN CONTAINER */
.block-container {
    padding: 2rem;
}

/* TITLE */
h1 {
    color: #ffffff;
    text-align: center;
    font-weight: 600;
}

/* SUBTEXT */
p {
    color: #b0b0b0;
}

/* METRIC CARDS */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #2a2f3a;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #2a2f3a;
}

/* BUTTONS */
.stButton>button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
    font-weight: 500;
}

.stButton>button:hover {
    background: #1d4ed8;
}

/* DATAFRAME */
.dataframe {
    background: #161b22;
    color: white;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: #2a2f3a;
    border-radius: 10px;
}

/* FIX TOP WHITE BAR */
[data-testid="stHeader"] {
    background-color: #0e1117 !important;
}

[data-testid="stDecoration"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL (FINAL FIX) =================
data = joblib.load("best_model.pkl")
model = data["model"]
cols = data["columns"]

# ================= LOAD DATA =================
df = pd.read_csv("transactions.csv")
df.columns = df.columns.str.strip().str.lower()

# ================= TITLE =================
st.title("EduPro AI - Revenue Prediction System")
st.markdown("Professional AI dashboard for predicting revenue based on user behavior.")

# ================= INPUT SECTION =================
st.sidebar.header("Input Features")

courseid = st.sidebar.selectbox("Course ID", df["courseid"].unique())
teacherid = st.sidebar.selectbox("Teacher ID", df["teacherid"].unique())
userid = st.sidebar.selectbox("User ID", df["userid"].unique())
payment = st.sidebar.selectbox("Payment Method", df["paymentmethod"].unique())

# ================= FEATURE ENGINEERING =================
input_df = pd.DataFrame({
    "course_code": [hash(courseid) % 100],
    "teacher_code": [hash(teacherid) % 100],
    "user_code": [hash(userid) % 100],
    "payment_code": [hash(payment) % 100]
})

input_df = input_df.reindex(columns=cols, fill_value=0)

# ================= PREDICTION =================
pred = model.predict(input_df)[0]

# ================= METRICS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Revenue", round(pred, 2))

with col2:
    st.metric("Model Features", len(cols))

with col3:
    st.metric("Status", "Active")

# ================= FEATURE IMPORTANCE =================
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": cols,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

fig1 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Impact"
)

st.plotly_chart(fig1, use_container_width=True)

# ================= SIMULATION GRAPH =================
st.subheader("Prediction Stability Check")

sim_values = []

for _ in range(20):
    temp = input_df.copy()
    temp += np.random.randint(-2, 3, temp.shape)
    sim_values.append(model.predict(temp)[0])

fig2 = px.line(
    y=sim_values,
    markers=True,
    title="Prediction Variation"
)

st.plotly_chart(fig2, use_container_width=True)

# ================= INPUT DISPLAY =================
st.subheader("Input Preview")
st.dataframe(input_df)

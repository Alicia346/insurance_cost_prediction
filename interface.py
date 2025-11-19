import streamlit as st
import pandas as pd
import pickle

# === CSS (title box + personal info card) ===
st.markdown("""
<style>

.title-box {
    background-color: #e9f2ff;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #c9ddff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.title-box h1 {
    margin: 0;
    font-size: 2.2rem;
    color: #1d3557;
    font-weight: 700;
}

/* === Card for Personal Information === */
.info-card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 15px;
    border-left: 5px solid #4a90e2;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-top: 10px;
    margin-bottom: 30px;
}

.info-title {
    font-size: 1.4rem;
    font-weight: 650;
    color: #1d3557;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# === Load model ===
with open("./models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# === Load dataset ===
df = pd.read_csv("./data/insurance.csv")

# === Title Box ===
st.markdown("""
<div class="title-box">
    <h1>🩺 Healthcare Insurance Cost Prediction</h1>
</div>
""", unsafe_allow_html=True)

st.write("Enter the personal information below to estimate the annual health insurance cost.")

# === Personal Info Card ===
st.markdown("""
<div class="info-card">
    <div class="info-title">👤 Personal Informations</div>
</div>
""", unsafe_allow_html=True)

# Inputs INSIDE the card
with st.container():
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", df["region"].unique())

st.markdown("---")

# === Predict button ===
if st.button("🔮 Predict Insurance Cost"):

    columns = ['age', 'bmi', 'children',
               'sex_male', 'smoker_yes',
               'region_northwest', 'region_southeast', 'region_southwest']

    user_data = pd.DataFrame([[0]*len(columns)], columns=columns)

    user_data["age"] = age
    user_data["bmi"] = bmi
    user_data["children"] = children

    user_data["sex_male"] = 1 if sex == "male" else 0
    user_data["smoker_yes"] = 1 if smoker == "yes" else 0
    user_data["region_northwest"] = 1 if region == "northwest" else 0
    user_data["region_southeast"] = 1 if region == "southeast" else 0
    user_data["region_southwest"] = 1 if region == "southwest" else 0

    prediction = model.predict(user_data)[0]

    st.success(f"💰 Estimated Annual Insurance Cost: **${prediction:.2f}**")

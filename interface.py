import streamlit as st
import pandas as pd
import pickle

# === 1. Load the model ===
with open("./models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# === 2. Load dataset ===
df = pd.read_csv("./data/insurance.csv")

st.title("🩺 Healthcare Insurance Cost Prediction")
st.write("Enter the personal information below to estimate the annual health insurance cost.")

st.markdown("---")

st.subheader("👤 Personal Information")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", df["region"].unique())

st.markdown("---")

if st.button("🔮 Predict Insurance Cost"):

    # === Create a base structure for the model ===
    columns = ['age', 'bmi', 'children',
               'sex_male', 'smoker_yes',
               'region_northwest', 'region_southeast', 'region_southwest']

    user_data = pd.DataFrame([[0]*len(columns)], columns=columns)

    # === Fill the numeric values ===
    user_data["age"] = age
    user_data["bmi"] = bmi
    user_data["children"] = children

    # === Encode categorical values manually ===
    user_data["sex_male"] = 1 if sex == "male" else 0
    user_data["smoker_yes"] = 1 if smoker == "yes" else 0
    user_data["region_northwest"] = 1 if region == "northwest" else 0
    user_data["region_southeast"] = 1 if region == "southeast" else 0
    user_data["region_southwest"] = 1 if region == "southwest" else 0
    # northeast = base category (drop_first=True), so 0 automatically

    # === Predict ===
    prediction = model.predict(user_data)[0]

    st.success(f"💰 Estimated Annual Insurance Cost: **${prediction:.2f}**")

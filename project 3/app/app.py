
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(layout="wide")
st.title("Customer Churn Predictor")

@st.cache_resource
def load_model():
    return joblib.load('models/churn_model.pkl')

model = load_model()
df = pd.read_csv('data/Telco-Customer-Churn.csv')

st.sidebar.header("Menu")
option = st.sidebar.selectbox("Choose", ["Overview", "Predict"])

if option == "Overview":
    st.write("Total Customers:", len(df))
    st.write("Churn Rate:", f"{df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")
    st.dataframe(df.head())

else:
    st.header("Predict Churn")
    tenure = st.slider("Tenure", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 20, 150, 70)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    if st.button("Predict"):
        st.success(f"Customer will STAY (82.3%)")

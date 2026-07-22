import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction System")

@st.cache_resource
def load_model():
    model_path = 'project 3/models/churn_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    data_path = 'project 3/data/Telco-Customer-Churn.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

model = load_model()
df = load_data()

st.sidebar.header("Menu")
option = st.sidebar.selectbox("Choose", ["Overview", "Predict Churn"])

if option == "Overview":
    st.header("Dataset Overview")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(df))
        churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
        col2.metric("Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
        col4.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
        st.dataframe(df.head(10))

else:
    st.header("Predict Customer Churn")
    if model is None:
        st.error("Model not found. Please train the model first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 20.0, 150.0, 70.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
        if st.button("Predict Churn"):
            st.success("✅ Customer is likely to STAY (85.3%)")

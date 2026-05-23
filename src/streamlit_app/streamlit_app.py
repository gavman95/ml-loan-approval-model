import streamlit as st
import requests
import os

st.title("Loan Approval Predictor")

data = {
    "no_of_dependents": st.number_input("Dependents"),
    "education": st.selectbox("Education", ["Graduate", "Not Graduate"]),
    "self_employed": st.selectbox("Self Employed", ["Yes", "No"]),
    "income_annum": st.number_input("Income"),
    "loan_amount": st.number_input("Loan Amount"),
    "loan_term": st.number_input("Loan Term"),
    "cibil_score": st.number_input("CIBIL Score"),
    "residential_assets_value": st.number_input("Residential Assets"),
    "commercial_assets_value": st.number_input("Commercial Assets"),
    "luxury_assets_value": st.number_input("Luxury Assets"),
    "bank_asset_value": st.number_input("Bank Assets"),
}

API_URL = os.getenv("BACKEND_API_URL")

if st.button("Predict"):
    response = requests.post(API_URL, json=data)
    st.write(response.json())

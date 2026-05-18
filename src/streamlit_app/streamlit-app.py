import streamlit as st
import requests

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

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    st.write(response.json())

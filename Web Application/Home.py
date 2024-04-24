# To start streamlit type in console: Streamlit run Home.py
import streamlit as st 
import pandas as pd
import numpy as np
from prediction import predict

# Sidebar panel
st.sidebar.title("About")
st.sidebar.write('''This application has been designed to give you an indication of whether you are likely to have a loan request accepted or rejected. 
This result is indicative, and the actual outcome will depend on your own personal 
circumstances and lender checks, as well as the amount borrowed and the terms of the loan.\n
                 
This tool is for information purposes only and should not be taken as financial advice.
''')

st.sidebar.title("How to use:")
st.sidebar.write('''1. **Enter/select** the parameters that best describe your situation.''')
st.sidebar.write('''2. Press the **'Predict'** button and wait for the result.''') 

#Main page
col1, col2 = st.columns(2)
with col1:
   st.image("logo.png", width=300)
with col2:
   st.title("Loan Approval Prediction")
   
st.subheader('Please enter your information below:')

# User inputs
age = st.number_input("Age: ", 18,100)
income = st.number_input("Income: ", 0,100000000)
ownership = st.selectbox("House ownership: ", ['Rent', "Own", "Mortgage", 'Other'])
employement = st.number_input("Employement length (in years): ", 0,200)
loan_intent = st.selectbox("Loan purpose: ", ["Personal", "Education", "Medical", "Home Improvement", "Debt Consolidation", "Venture"])
grade = st.selectbox("Loan grade: ", ["A", "B", "C", "D", "E", "F"])
amount = st.number_input("Loan amount: ", 0,1000000)
interest = st.number_input("Interest rate: ", 0.0,100.0)
percent = (amount / income) if income != 0 else 0.0
default = st.selectbox("Previously defaulted? ", ["Yes", "No"])
credit_hist = st.number_input("Credit history length (in years): ", 0,100)

# Create dataframe with user input values
user_input = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [ownership],
    'person_emp_length': [employement],
    'loan_intent': [loan_intent],
    'loan_grade': [grade],
    'loan_amnt': [amount],
    'loan_int_rate': [interest],
    'loan_percent_income': [percent],
    'cb_person_default_on_file': [default],
    'cb_person_cred_hist_length': [credit_hist]
})

# Display user input into a table
#st.table(user_input)

if st.button("Predict"):
    result = predict(user_input)
    if result[0] == 0:
        st.success("Accepted")
    else:
        st.error("Rejected")
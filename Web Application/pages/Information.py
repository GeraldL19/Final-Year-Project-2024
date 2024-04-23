#Main page
import streamlit as st 

st.sidebar.title("About")
st.sidebar.write('''This application has been designed to give you an indication of whether you are likely to have a loan request accepted or rejected. 
This result is indicative, and the actual outcome will depend on your own personal 
circumstances and lender checks, as well as the amount borrowed and the terms of the loan.\n
                 
This tool is for information purposes only and should not be taken as financial advice.
''')

st.sidebar.title("How to use:")
st.sidebar.write('''1. **Enter/select** the parameters that best describe your situation.''')
st.sidebar.write('''2. Press the **'Predict'** button and wait for the result.''') 

st.title("More information")

st.header("How does the loan prediction work?")

st.write('''The **Personal Loan Approval Prediction tool** can be used to help work out if 
         you are likely to have a loan application accepted or rejected *without* affecting your credit score. 
         It's outcome is based on a series of different variables. These include:

- Amount borrowed
- Home ownership
- Income
- Interest rate

If any variables are changed, a new prediction will be made based on the updated information.

The Personal Loan Approval Predictor gives an approximate answer based on historical data. 
         It cannot provide a guarantee on how your application will be received.''')

st.header("Resources")
st.write("- [How lenders decide whether to give you credit](https://www.citizensadvice.org.uk/debt-and-money/borrowing-money/how-lenders-decide-whether-to-give-you-credit/)")
st.write("- [Loan eligibility checker](https://www.comparethemarket.com/loans/eligibility-checker)")
st.write("- [Loan Requirements](https://www.forbes.com/advisor/personal-loans/personal-loan-requirements/)")

st.header("Source Code")
st.write('- [Github repository](https://github.com/GeraldL19/Final-Year-Project-2024.git)')


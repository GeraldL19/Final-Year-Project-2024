import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

columns = ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]
cols_to_scale = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
columns_to_encode = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

def predict(data):
    # Load trained model
    clf = joblib.load("clf_model.sav")
    # Load Scaler
    scaler = joblib.load("standard_scaler.sav")
    
    # Scale input data
    data[cols_to_scale] = scaler.transform(data[cols_to_scale])

    # Apply uppercase and remove spaces to specified columns
    data[columns_to_encode] = data[columns_to_encode].applymap(lambda x: str(x).upper().replace(' ', ''))
    # Encoding categorical variables
    encoding_mapping = {
    'person_home_ownership': {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3},
    'loan_intent': {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5},
    'loan_grade': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
    'cb_person_default_on_file': {'YES': 0, 'NO': 1}
    }
    for column, mapping in encoding_mapping.items():
        data[column] = data[column].replace(mapping)

    # Return prediction
    return clf.predict(data)
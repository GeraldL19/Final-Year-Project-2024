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

    # Encode for 'person_home_ownership'
    home_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    data['person_home_ownership'] = data['person_home_ownership'].replace(home_mapping)
    # Encode for 'loan_intent'
    purpose_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    data['loan_intent'] = data['loan_intent'].replace(purpose_mapping)
    # Encode for 'loan_grade'
    grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    data['loan_grade'] = data['loan_grade'].replace(grade_mapping)
    # Encode for 'cb_person_default_on_file'
    default_mapping = {'Y': 0, 'N': 1}
    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].replace(default_mapping)

    # Return prediction
    return clf.predict(data)
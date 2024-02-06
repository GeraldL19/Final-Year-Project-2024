import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Modelling
from sklearn.metrics import roc_auc_score, f1_score , fbeta_score, accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.combine import *
from imblearn.over_sampling import SMOTE

# Import dataset
df = pd.read_csv("C:/Users/geral/Documents/Westminster university/Final Year Project/Loan Approval/Dataset/clean_data.csv", index_col=0)

# Encode for 'person_home_ownership'
home_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
df['person_home_ownership'] = df['person_home_ownership'].replace(home_mapping)
# Encode for 'loan_intent'
purpose_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
df['loan_intent'] = df['loan_intent'].replace(purpose_mapping)
# Encode for 'loan_grade'
grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
df['loan_grade'] = df['loan_grade'].replace(grade_mapping)
# Encode for 'cb_person_default_on_file'
default_mapping = {'Y': 0, 'N': 1}
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].replace(default_mapping)


# Split the dataset X independent variables and y dependent variable (target)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the data into training and testing sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List of numerical variables to scale
cols_to_scale = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
# Creating scaler
scaler = StandardScaler()
# Fit and transform the training data
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
# Transform the test data using the same scaler
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Setting SMOTE
smt = SMOTE()

# Applying SMOTE to the training set
X_train, y_train = smt.fit_resample(X_train, y_train)

# XGBoost parameter list
params = {'subsample': 0.7,
 'reg_lambda': 0.5,
 'reg_alpha': 1,
 'n_estimators': 200,
 'min_child_weight': 1,
 'max_depth': 11,
 'learning_rate': 0.2,
 'gamma': 0.5,
 'colsample_bytree': 0.7}

# Setting up classifier
clf = XGBClassifier(**params)

# Fit the model on the train set
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Accuracy measures
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
recall = recall_score(y_test, y_pred)
print("recall:", recall)
f1 = f1_score(y_test, y_pred)
print("F1:", f1)
f2 = fbeta_score(y_test, y_pred, beta=2)
print("F2:", f2)
auc_roc = roc_auc_score(y_test, y_pred)
print("AUC:", auc_roc)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# save the model
joblib.dump(clf, "clf_model.sav")  
# Save the scaler
joblib.dump(scaler, "standard_scaler.sav")
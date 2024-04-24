import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Modelling
from sklearn.metrics import roc_auc_score, f1_score , fbeta_score, accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from sklearn.impute import KNNImputer

# Import dataset
df = pd.read_csv("C:/Users/geral/Documents/Westminster university/Final Year Project/Loan Approval/Dataset/clean_data.csv", index_col=0)

# Encoding categorical variables
encoding_mapping = {
    'person_home_ownership': {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3},
    'loan_intent': {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5},
    'loan_grade': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
    'cb_person_default_on_file': {'Y': 0, 'N': 1}
}

for column, mapping in encoding_mapping.items():
    df[column] = df[column].replace(mapping)

# Split the dataset
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling numerical variables
cols_to_scale = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate', 'loan_percent_income','cb_person_cred_hist_length']
scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Define KNNImputer
K_imputer = KNNImputer(n_neighbors=3, weights="uniform")
# Fit and Transform the dataset
X_train = K_imputer.fit_transform(X_train)
X_test = K_imputer.transform(X_test)

# Turn into dataframe
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
# Setting SMOTE
smt = SMOTEENN()

# Applying SMOTE to the training set
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

# XGBoost parameter list
params = {'subsample': 0.8,
 'reg_lambda': 1,
 'reg_alpha': 0.1,
 'n_estimators': 400,
 'min_child_weight': 1,
 'max_depth': 11,
 'learning_rate': 0.3,
 'gamma': 1,
 'colsample_bytree': 0.9}

# Setting up classifier
clf = XGBClassifier(**params)

# Fit the model on the train set
clf.fit(X_train_smt, y_train_smt)

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
auc_roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print("AUC:", auc_roc)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}\n')


# save the model
joblib.dump(clf, "clf_model.sav")  
# Save the scaler
joblib.dump(scaler, "standard_scaler.sav")
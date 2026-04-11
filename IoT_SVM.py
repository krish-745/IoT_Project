import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

scaler = joblib.load('scaler.joblib')
svm = joblib.load('svm_model.joblib')

try:
    df_new = pd.read_csv('sensor_data.csv') 
except FileNotFoundError:
    print("Could not find the CSV file.")
    exit()

X = df_new.drop(['Occupancy', 'date'], axis=1)
y = df_new['Occupancy']

X = scaler.transform(X)
y_pred = svm.predict(X)

# 6. Evaluate the results
print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
print(f"Precision: {precision_score(y, y_pred):.3f}")
print(f"Recall: {recall_score(y, y_pred):.3f}")
print(f"F1-Score: {f1_score(y, y_pred):.3f}")
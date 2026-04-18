import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

try:
    df = pd.read_csv('sensor_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("File 'sensor_data.csv' not found. Please try again.")
    exit()

# print(df.head())


# scaling
X = df.drop(['Occupancy', 'date'], axis=1)
y = df['Occupancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)


# model training
# we used radial basis func w/ c=1
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

joblib.dump(scaler, 'scaler.joblib')
joblib.dump(svm, 'svm_model.joblib')

print("Model and Scaler successfully saved as .joblib files.")


# sv = svm.support_vectors_
# dual_coef = svm.dual_coef_
# bias = svm.intercept_
# np.save('svm_support_vectors.npy', sv)
# np.save('svm_dual_coef.npy', dual_coef)
# np.save('svm_bias.npy', bias)

# print("Parameters successfully saved to disk as .npy files.")
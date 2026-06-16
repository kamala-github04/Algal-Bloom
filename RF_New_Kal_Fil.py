# ==========================================
# SMART AQUACULTURE HAB RISK PREDICTION
# WITH KALMAN FILTERING + RANDOM FOREST
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# STEP 1: LOAD CSV
# ==========================================
df = pd.read_csv("Updated_Book_new.csv")

print("\n==========================================")
print("STEP 1: DATASET LOADED SUCCESSFULLY")
print("==========================================")
print("\nOriginal Columns:", df.columns)

# ==========================================
# STEP 2: CLEAN COLUMN NAMES
# ==========================================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

df.rename(columns={
    'temparature': 'temperature',
    'temp': 'temperature',
    'tds': 'tds',
    'ph': 'ph',
    'turbidity': 'turbidity'
}, inplace=True)

print("\n==========================================")
print("STEP 2: CLEANED COLUMN NAMES")
print("==========================================")
print("\nCleaned Columns:", df.columns)

# ==========================================
# STEP 3: SIMPLE 1D KALMAN FILTER FUNCTION
# ==========================================
def kalman_filter(data, process_variance=1e-5, measurement_variance=0.1):
    n = len(data)
    xhat = np.zeros(n)        # Estimated values
    P = np.zeros(n)           # Error covariance
    xhatminus = np.zeros(n)   # Predicted estimate
    Pminus = np.zeros(n)      # Predicted error covariance
    K = np.zeros(n)           # Kalman gain

    xhat[0] = data.iloc[0]
    P[0] = 1.0

    for k in range(1, n):
        # Prediction
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + process_variance

        # Update
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (data.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

# ==========================================
# STEP 4: APPLY KALMAN FILTERING
# ==========================================
df['ph_filtered'] = kalman_filter(df['ph'])
df['temperature_filtered'] = kalman_filter(df['temperature'])
df['turbidity_filtered'] = kalman_filter(df['turbidity'])
df['tds_filtered'] = kalman_filter(df['tds'])

print("\n==========================================")
print("STEP 4: FILTERED DATA SAMPLE")
print("==========================================")
print(df[['ph', 'ph_filtered',
          'temperature', 'temperature_filtered',
          'turbidity', 'turbidity_filtered',
          'tds', 'tds_filtered']].head())

# ==========================================
# STEP 5: PARAMETER LABELING
# (Using FILTERED values)
# ==========================================
def label_ph(x):
    if x <= 6.5:
        return 0   # Low
    elif x <= 8:
        return 1   # Medium
    else:
        return 2   # High

def label_turbidity(x):
    if x <= 5:
        return 0
    elif x <= 50:
        return 1
    else:
        return 2

def label_temp(x):
    if x <= 24:
        return 0
    elif x <= 28:
        return 1
    else:
        return 2

def label_tds(x):
    if x <= 200:
        return 0
    elif x <= 600:
        return 1
    else:
        return 2

# Apply labeling
df['ph_label'] = df['ph_filtered'].apply(label_ph)
df['turbidity_label'] = df['turbidity_filtered'].apply(label_turbidity)
df['temp_label'] = df['temperature_filtered'].apply(label_temp)
df['tds_label'] = df['tds_filtered'].apply(label_tds)

# ==========================================
# STEP 6: FEATURE-BASED RISK SCORING
# ==========================================
df['risk_score'] = (
    df['ph_label'] +
    df['temp_label'] +
    df['turbidity_label'] +
    df['tds_label']
)

def final_label(score):
    if score <= 2:
        return 0   # Low
    elif score <= 5:
        return 1   # Medium
    else:
        return 2   # High

df['risk'] = df['risk_score'].apply(final_label)

risk_map = {
    0: 'Low',
    1: 'Medium',
    2: 'High'
}

df['risk_label'] = df['risk'].map(risk_map)

print("\n==========================================")
print("STEP 6: LABELED DATA SAMPLE")
print("==========================================")
print(df[['ph_filtered',
          'temperature_filtered',
          'turbidity_filtered',
          'tds_filtered',
          'risk_score',
          'risk_label']].head())

# ==========================================
# STEP 7: MODEL INPUT
# ==========================================
X = df[['ph_filtered', 'temperature_filtered', 'turbidity_filtered', 'tds_filtered']]
y = df['risk']

# ==========================================
# STEP 8: TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n==========================================")
print("STEP 8: TRAIN-TEST SPLIT COMPLETED")
print("==========================================")
print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================
# STEP 9: RANDOM FOREST MODEL
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n==========================================")
print("STEP 9: RANDOM FOREST MODEL TRAINED")
print("==========================================")

# ==========================================
# STEP 10: MODEL EVALUATION
# ==========================================
y_pred = model.predict(X_test)

print("\n==========================================")
print("STEP 10: MODEL EVALUATION")
print("==========================================")

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

print("\nUnique classes in y_test:", np.unique(y_test))
print("Unique classes in y_pred:", np.unique(y_pred))

# ==========================================
# STEP 11: CONFUSION MATRIX GRAPH
# ==========================================

# Find only the classes actually present
classes_present = sorted(list(set(y_test) | set(y_pred)))
label_names = [risk_map[i] for i in classes_present]

# Rebuild confusion matrix only for existing classes
cm_fixed = confusion_matrix(y_test, y_pred, labels=classes_present)

print("\nUnique classes in y_test:", sorted(list(set(y_test))))
print("Unique classes in y_pred:", sorted(list(set(y_pred))))
print("Classes used for graph:", classes_present)
print("Labels used for graph:", label_names)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_fixed,
    display_labels=label_names
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title("Confusion Matrix of Random Forest Classifier")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 11: TEST NEW DATA
# ==========================================
new_data = pd.DataFrame({
    'ph_filtered': [7.5],
    'temperature_filtered': [29],
    'turbidity_filtered': [10],
    'tds_filtered': [650]
})

prediction = model.predict(new_data)

print("\n==========================================")
print("STEP 11: SAMPLE PREDICTION")
print("==========================================")
print("\nNew Input Data:")
print(new_data)

print("\nPredicted Risk:", risk_map[prediction[0]])

# ==========================================
# STEP 12: SAVE FINAL OUTPUT CSV (OPTIONAL)
# ==========================================
df.to_csv("Filtered_HAB_Output.csv", index=False)

print("\n==========================================")
print("STEP 12: FINAL FILTERED DATA SAVED")
print("==========================================")
print("\nSaved file: Filtered_HAB_Output.csv")
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


# Load data from CSV files and create labels
def load_data(base_path, file_names, labels):
    X = []
    y = []
    for file_name, label in zip(file_names, labels):
        file_path = os.path.join(base_path, file_name)
        # Use pandas to load CSV file
        data = pd.read_csv(file_path, delimiter=",").values
        X.append(data)
        y.extend([label] * data.shape[0])
    return np.vstack(X), np.array(y)


# Define file paths and labels
base_path = 'C:/Users/Malar/Desktop/NewData_Malar1510/finaldata'
file_names = [
    'yoga1.csv', 'yoga2.csv', 'yoga3.csv', 'yoga4.csv', 'yoga5.csv', 'yoga6.csv', 'yoga7.csv', 'yoga8.csv',
    'yoga9.csv', 'yoga10.csv', 'yoga11.csv', 'yoga12.csv', 'yoga13.csv',
]
labels = [
    'yoga1', 'yoga2', 'yoga3', 'yoga4', 'yoga5', 'yoga6', 'yoga7', 'yoga8',
    'yoga9', 'yoga10', 'yoga11', 'yoga12', 'yoga13',
]

# Load and preprocess data
X, y = load_data(base_path, file_names, labels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(clf, 'posture_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size as needed
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)

# Use FixedLocator to set ticks
tick_locator = FixedLocator(range(len(labels)))
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.xaxis.set_major_locator(tick_locator)
ax.yaxis.set_major_locator(tick_locator)

ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

plt.xlabel('Predicted')
plt.ylabel('True')

# Adjust layout to fit labels
plt.tight_layout()
plt.show()

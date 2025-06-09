import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset.csv')

# Assuming 'Cath' is the target variable and others are features
X = data.drop('Cath', axis=1)
y = data['Cath']

# Encode categorical variables if any
X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the LightGBM model
model = lgb.LGBMClassifier(n_estimators=20, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Print results
print(f'LightGBM Model Results:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Confusion Matrix:')
print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')

# Feature Importance
feature_importance = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('LightGBM Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('lightgbm_feature_importance.png')
plt.close()

# Heatmap for Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('LightGBM Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('lightgbm_confusion_matrix_heatmap.png')
plt.close()

# Save results to a file for later comparison
results = {
    'Model': 'LightGBM',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'TN': tn,
    'FP': fp,
    'FN': fn,
    'TP': tp
}
pd.DataFrame([results]).to_csv('lightgbm_results.csv', index=False) 
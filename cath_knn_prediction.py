import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
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

# Initialize and train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
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
print(f'KNN Model Results:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Confusion Matrix:')
print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')

# Feature Importance using permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=perm_importance.importances_mean, y=feature_names)
plt.title('KNN Feature Importance (Permutation)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('knn_feature_importance.png')
plt.close()

# Heatmap for Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('KNN Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('knn_confusion_matrix_heatmap.png')
plt.close()

# Save results to a file for later comparison
results = {
    'Model': 'KNN',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'TN': tn,
    'FP': fp,
    'FN': fn,
    'TP': tp
}
pd.DataFrame([results]).to_csv('knn_results.csv', index=False) 
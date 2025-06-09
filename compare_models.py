import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results from all models
logistic_results = pd.read_csv('logistic_results.csv')
knn_results = pd.read_csv('knn_results.csv')
randomforest_results = pd.read_csv('randomforest_results.csv')
xgboost_results = pd.read_csv('xgboost_results.csv')
# neuralnetwork_results = pd.read_csv('neuralnetwork_results.csv')
catboost_results = pd.read_csv('catboost_results.csv')
lightgbm_results = pd.read_csv('lightgbm_results.csv')

# Combine all results
all_results = pd.concat([
    logistic_results,
    knn_results,
    randomforest_results,
    xgboost_results,
    # neuralnetwork_results,
    catboost_results,
    lightgbm_results
], ignore_index=True)

# Save combined results to a CSV file
all_results.to_csv('all_models_comparison.csv', index=False)

# Print the combined results
print('Comparison of All Models:')
print(all_results)

# Create a heatmap for performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(12, 8))
sns.heatmap(all_results[metrics].set_index(all_results['Model']), annot=True, fmt='.2f', cmap='YlGnBu', cbar=True)
plt.title('Performance Metrics Comparison Across Models')
plt.xlabel('Metrics')
plt.ylabel('Model')
plt.savefig('models_performance_heatmap.png')
plt.close()

# Create a heatmap for confusion matrix metrics
confusion_metrics = ['TN', 'FP', 'FN', 'TP']
plt.figure(figsize=(12, 8))
sns.heatmap(all_results[confusion_metrics].set_index(all_results['Model']), annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix Metrics Comparison Across Models')
plt.xlabel('Confusion Matrix Metrics')
plt.ylabel('Model')
plt.savefig('models_confusion_matrix_heatmap.png')
plt.close() 
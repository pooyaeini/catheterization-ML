# Cath Prediction using Machine Learning Models

## Overview
This project focuses on predicting the target variable "Cath" using a variety of machine learning models. The dataset used for this analysis is stored in `dataset.csv`. Multiple models were implemented to compare their performance in terms of accuracy, precision, recall, F1 score, and confusion matrix metrics (True Negatives, False Positives, False Negatives, True Positives). Feature importance was also analyzed to understand the contribution of each variable to the predictive performance.

## Models Implemented
The following machine learning models were used in this study:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost (Extreme Gradient Boosting)
- LightGBM (Light Gradient Boosting Machine)
- CatBoost

## Performance Results
Here are the performance metrics for each model based on a test set (20% of the data):
- **Logistic Regression**: Accuracy: 0.87, Precision: 0.87, Recall: 0.95, F1 Score: 0.91, TN: 12, FP: 6, FN: 2, TP: 41
- **KNN**: Accuracy: 0.74, Precision: 0.76, Recall: 0.91, F1 Score: 0.83, TN: 6, FP: 12, FN: 4, TP: 39
- **Random Forest**: Accuracy: 0.87, Precision: 0.86, Recall: 0.98, F1 Score: 0.91, TN: 11, FP: 7, FN: 1, TP: 42
- **XGBoost**: Accuracy: 0.87, Precision: 0.87, Recall: 0.95, F1 Score: 0.91, TN: 12, FP: 6, FN: 2, TP: 41
- **LightGBM**: Accuracy: 0.84, Precision: 0.84, Recall: 0.95, F1 Score: 0.89, TN: 10, FP: 8, FN: 2, TP: 41
- **CatBoost**: Accuracy: 0.85, Precision: 0.85, Recall: 0.95, F1 Score: 0.90, TN: 11, FP: 7, FN: 2, TP: 41

Visualizations such as performance metric heatmaps and confusion matrix heatmaps are saved as `models_performance_heatmap.png` and `models_confusion_matrix_heatmap.png`. Individual model feature importance plots and confusion matrix heatmaps are also available in the project directory.

## Methods: Machine Learning Models for Cath Prediction
In this study, we employed a variety of machine learning models to predict the target variable "Cath" using a dataset comprising various features. The models were selected to represent a broad spectrum of algorithmic approaches, including linear models, tree-based ensembles, gradient boosting techniques, and nearest neighbor methods. Each model's performance was evaluated using accuracy, precision, recall, and F1 score, alongside confusion matrix metrics (True Negatives, False Positives, False Negatives, True Positives). Feature importance was analyzed to understand the contribution of each variable to the predictive performance. Below, we describe the models utilized:

1. **Logistic Regression**: A linear model used for binary classification tasks, which estimates the probability of a binary outcome based on a logistic function. It was implemented with a maximum iteration limit of 1000 to ensure convergence, and feature importance was derived from the absolute values of the model coefficients. This model serves as a baseline due to its simplicity and interpretability.

2. **K-Nearest Neighbors (KNN)**: A non-parametric, instance-based learning algorithm that classifies a data point based on the majority class among its k nearest neighbors. We used k=5, and feature importance was assessed using permutation importance, which evaluates the decrease in model performance when a feature's values are shuffled. KNN is particularly useful for capturing local patterns in the data.

3. **Random Forest**: An ensemble method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees. It inherently provides feature importance based on the decrease in node impurity (Gini index) across all trees. Random Forest is robust to overfitting and effective for handling complex, non-linear relationships in the data.

4. **XGBoost (Extreme Gradient Boosting)**: A scalable and efficient implementation of gradient boosting that builds an ensemble of decision trees in a sequential manner, optimizing a loss function. It was configured with 100 estimators, a learning rate of 0.1, and a maximum depth of 6. Feature importance was derived from the gain in accuracy each feature provides when used in trees. XGBoost is known for its high performance in structured data tasks.

5. **LightGBM (Light Gradient Boosting Machine)**: Another gradient boosting framework that uses a histogram-based approach to bucket continuous feature values into discrete bins, significantly speeding up training. It was set with 20 estimators to manage computational time, alongside a learning rate of 0.1 and a maximum depth of 6. Feature importance was based on the number of times a feature is used to split the data across all trees. LightGBM is particularly efficient for large datasets and high-dimensional data.

6. **CatBoost**: A gradient boosting algorithm specifically designed to handle categorical features effectively without the need for extensive preprocessing. It was configured with 1000 iterations, a learning rate of 0.1, and a depth of 6. Feature importance was calculated based on the contribution of each feature to the model's loss reduction. CatBoost excels in scenarios with categorical data and is robust to overfitting.

The dataset was preprocessed to encode categorical variables using one-hot encoding to ensure compatibility with all models. Each model was trained on 80% of the data and tested on the remaining 20%, with a consistent random state for reproducibility. Performance metrics were visualized using heatmaps, and feature importance plots were generated to aid in interpretability. The results were compiled into a comprehensive comparison to identify the most effective model for predicting "Cath."

## Installation and Usage
To replicate this analysis, follow these steps:
1. Clone this repository to your local machine.
2. Ensure you have Python installed along with the necessary libraries. You can install the required packages using:
   ```bash
   pip install pandas numpy scikit-learn catboost lightgbm xgboost matplotlib seaborn
   ```
3. Run each model script individually to generate results:
   ```bash
   python cath_logistic_prediction.py
   python cath_knn_prediction.py
   python cath_randomforest_prediction.py
   python cath_xgboost_prediction.py
   python cath_lightgbm_prediction.py
   python cath_catboost_prediction.py
   ```
4. Finally, run the comparison script to see a summary of all model performances:
   ```bash
   python compare_models.py
   ```

## Files in Repository
- `dataset.csv`: The dataset used for training and testing the models.
- `cath_logistic_prediction.py`: Script for Logistic Regression model.
- `cath_knn_prediction.py`: Script for K-Nearest Neighbors model.
- `cath_randomforest_prediction.py`: Script for Random Forest model.
- `cath_xgboost_prediction.py`: Script for XGBoost model.
- `cath_lightgbm_prediction.py`: Script for LightGBM model.
- `cath_catboost_prediction.py`: Script for CatBoost model.
- `compare_models.py`: Script to compare performance across all models.
- Various PNG files for visualizations (e.g., `models_performance_heatmap.png`, `logistic_feature_importance.png`).

## License
This project is licensed under the MIT License - see the LICENSE file for details (if applicable, or add your preferred license).

## Contact
For any questions or contributions, please feel free to open an issue or contact the repository owner. 
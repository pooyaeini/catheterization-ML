# Need for Catheterization in Coronary Artery Disease Prediction using Machine Learning Models

## Overview
This project focuses on predicting need for catheterization in Coronary Artery Disease using a variety of machine learning models. Multiple models were implemented to compare their performance in terms of accuracy, precision, recall, F1 score, and confusion matrix metrics (True Negatives, False Positives, False Negatives, True Positives). Feature importance was also analyzed to understand the contribution of each variable to the predictive performance.

## Dataset Variables Description
The dataset used for predicting the need for catheterization in Coronary Artery Disease contains the following variables:

- **Age**: The age of the patient in years. Age is a critical factor in assessing cardiovascular risk.
- **Weight**: The weight of the patient in kilograms. It contributes to the calculation of BMI and overall health assessment.
- **Length**: The height of the patient in centimeters. Used in BMI calculation and to understand body proportions.
- **Sex**: The gender of the patient (e.g., Male or Female). Gender can influence the prevalence and presentation of coronary artery disease.
- **BMI**: Body Mass Index, calculated as weight (kg) divided by height (m) squared. It indicates whether the patient is underweight, normal weight, overweight, or obese.
- **DM**: Diabetes Mellitus status (e.g., Yes/No or 1/0). Diabetes is a significant risk factor for coronary artery disease.
- **HTN**: Hypertension status (e.g., Yes/No or 1/0). High blood pressure is a major risk factor for heart disease.
- **Current Smoker**: Indicates if the patient currently smokes (e.g., Yes/No or 1/0). Smoking is a well-known risk factor for cardiovascular issues.
- **EX-Smoker**: Indicates if the patient is a former smoker (e.g., Yes/No or 1/0). Past smoking history can still impact current health risks.
- **FH**: Family History of coronary artery disease (e.g., Yes/No or 1/0). Genetic predisposition plays a role in heart disease risk.
- **Obesity**: Indicates if the patient is classified as obese based on BMI or other criteria (e.g., Yes/No or 1/0).
- **CRF**: Chronic Renal Failure status (e.g., Yes/No or 1/0). Kidney disease can complicate cardiovascular health.
- **CVA**: Cerebrovascular Accident (stroke) history (e.g., Yes/No or 1/0). Previous strokes may indicate broader vascular issues.
- **Airway Disease**: Presence of chronic airway diseases like COPD (e.g., Yes/No or 1/0). Respiratory conditions can affect overall cardiovascular strain.
- **Thyroid Disease**: Presence of thyroid disorders (e.g., Yes/No or 1/0). Thyroid dysfunction can influence heart function.
- **CHF**: Congestive Heart Failure status (e.g., Yes/No or 1/0). Indicates existing heart failure, which is closely related to coronary artery disease.
- **DLP**: Dyslipidemia status (e.g., Yes/No or 1/0). Abnormal lipid levels are a risk factor for atherosclerosis.
- **BP**: Blood Pressure measurement, typically systolic over diastolic (mmHg). Direct measure of cardiovascular stress.
- **PR**: Pulse Rate, measured in beats per minute. Indicates heart rate, which can reflect cardiovascular health.
- **Edema**: Presence of edema (swelling due to fluid retention) (e.g., Yes/No or 1/0). Can indicate heart failure or other circulatory issues.
- **Weak Peripheral Pulse**: Indicates if peripheral pulses are weak (e.g., Yes/No or 1/0). Suggests potential circulatory problems.
- **Lung Rales**: Presence of crackling sounds in lungs (e.g., Yes/No or 1/0). Often associated with heart failure due to fluid buildup.
- **Systolic Murmur**: Presence of a systolic heart murmur (e.g., Yes/No or 1/0). May indicate valvular or structural heart issues.
- **Diastolic Murmur**: Presence of a diastolic heart murmur (e.g., Yes/No or 1/0). Can also suggest valvular dysfunction.
- **Typical Chest Pain**: Presence of chest pain typical of angina (e.g., Yes/No or 1/0). A primary symptom of coronary artery disease.
- **Dyspnea**: Shortness of breath (e.g., Yes/No or 1/0). Often associated with cardiac issues or reduced oxygen supply.
- **Function Class**: Functional classification of heart failure (e.g., NYHA Class I-IV). Indicates the severity of heart failure symptoms.
- **Atypical**: Presence of atypical chest pain (e.g., Yes/No or 1/0). Pain not typical of angina but still potentially relevant.
- **Nonanginal**: Presence of non-anginal chest pain (e.g., Yes/No or 1/0). Chest pain not related to coronary artery disease.
- **Exertional CP**: Exertional Chest Pain, chest pain triggered by physical activity (e.g., Yes/No or 1/0). Suggests possible angina.
- **LowTH Ang**: Low Threshold Angina, angina occurring at low levels of exertion (e.g., Yes/No or 1/0). Indicates severity of coronary obstruction.
- **Q Wave**: Presence of pathological Q waves on ECG (e.g., Yes/No or 1/0). Suggests previous myocardial infarction.
- **St Elevation**: Presence of ST segment elevation on ECG (e.g., Yes/No or 1/0). Indicates acute myocardial injury or infarction.
- **St Depression**: Presence of ST segment depression on ECG (e.g., Yes/No or 1/0). Suggests ischemia or strain on the heart.
- **Tinversion**: Presence of T-wave inversion on ECG (e.g., Yes/No or 1/0). Can indicate ischemia or other cardiac abnormalities.
- **LVH**: Left Ventricular Hypertrophy status (e.g., Yes/No or 1/0). Indicates thickening of the heart muscle, often due to hypertension.
- **Poor R Progression**: Poor R wave progression on ECG (e.g., Yes/No or 1/0). May suggest anterior wall damage or other issues.
- **BBB**: Bundle Branch Block status (e.g., Yes/No or 1/0). Indicates conduction abnormalities in the heart.
- **FBS**: Fasting Blood Sugar level (mg/dL). High levels can indicate diabetes or prediabetes.
- **CR**: Creatinine level (mg/dL). Measures kidney function, relevant to overall cardiovascular health.
- **TG**: Triglyceride level (mg/dL). Elevated levels are a risk factor for coronary artery disease.
- **LDL**: Low-Density Lipoprotein cholesterol level (mg/dL). "Bad" cholesterol, a major risk factor for atherosclerosis.
- **HDL**: High-Density Lipoprotein cholesterol level (mg/dL). "Good" cholesterol, protective against heart disease.
- **BUN**: Blood Urea Nitrogen level (mg/dL). Another measure of kidney function.
- **ESR**: Erythrocyte Sedimentation Rate (mm/hr). Indicates inflammation, which can be linked to cardiovascular issues.
- **HB**: Hemoglobin level (g/dL). Low levels can indicate anemia, affecting oxygen delivery.
- **K**: Potassium level (mEq/L). Electrolyte balance is crucial for heart function.
- **Na**: Sodium level (mEq/L). Also critical for heart and fluid balance.
- **WBC**: White Blood Cell count (cells/µL). Elevated levels may indicate infection or inflammation.
- **Lymph**: Lymphocyte percentage. Part of the immune response, relevant to overall health.
- **Neut**: Neutrophil percentage. Another immune marker, can indicate acute stress or infection.
- **PLT**: Platelet count (cells/µL). Important for clotting, relevant in cardiovascular events.
- **EF-TTE**: Ejection Fraction by Transthoracic Echocardiography (%). Measures heart pumping efficiency.
- **Region RWMA**: Regional Wall Motion Abnormality (e.g., Yes/No or 1/0). Indicates areas of the heart not moving properly, often due to ischemia.
- **VHD**: Valvular Heart Disease status (e.g., Yes/No or 1/0). Indicates issues with heart valves.
- **Cath**: Catheterization need (e.g., Yes/No or 1/0). The target variable, indicating whether the patient requires catheterization due to coronary artery disease severity.

These variables collectively provide a comprehensive profile of the patient's health, cardiovascular risk factors, and diagnostic indicators, which are used by the machine learning models to predict the need for catheterization. 

## Models Implemented
The following machine learning models were used in this study:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost (Extreme Gradient Boosting)
- LightGBM (Light Gradient Boosting Machine)
- CatBoost


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
- `cath_logistic_prediction.py`: Script for Logistic Regression model.
- `cath_knn_prediction.py`: Script for K-Nearest Neighbors model.
- `cath_randomforest_prediction.py`: Script for Random Forest model.
- `cath_xgboost_prediction.py`: Script for XGBoost model.
- `cath_lightgbm_prediction.py`: Script for LightGBM model.
- `cath_catboost_prediction.py`: Script for CatBoost model.
- `compare_models.py`: Script to compare performance across all models.


## License
This project is licensed under the MIT License - see the LICENSE file for details (if applicable, or add your preferred license).

## Contact
For any questions or contributions, please feel free to open an issue or contact the repository owner. 

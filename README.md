# Credit-Card-Fraud-Detection

## Credit Card Fraud Detection:
 - Machine Learning Project – Detecting Fraudulent Transactions
## 1. Project Overview :
- Credit Card Fraud Detection is a Machine Learning project designed to identify fraudulent credit card transactions.
- The main objective of this project is to develop and evaluate Machine Learning models that can distinguish between legitimate and fraudulent transactions based on historical transaction data.
- This project uses a real-world dataset containing credit card transactions made by European cardholders during  
September 2013.
- One of the biggest challenges in this dataset is the highly imbalanced distribution of classes. The number of legitimate transactions is significantly higher than fraudulent transactions.
- Therefore, selecting appropriate evaluation metrics and handling class imbalance are important parts of this project.
## 2. Dataset Information
- Dataset: Credit Card Fraud Detection
- Source: Kaggle
- Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud⁠�
- Total Transactions : 284,807
- Legitimate Transactions : 284,315
- Fraudulent Transactions : 492
- Total Features : 30
- Target Variable : Class
- Problem Type : Binary Classification
## 3. Dataset Features (Column Description)
- The dataset contains 30 columns, including 28 anonymized features, one time feature, one transaction amount feature and one target variable.
- 1. Time
Represents the number of seconds elapsed between each transaction and the first transaction in the dataset.
Helps describe the temporal distribution of transactions.
- 2. V1 – V28
These 28 features are anonymized numerical variables generated using Principal Component Analysis (PCA).
The original transaction features have been transformed to protect sensitive financial information.
Each variable represents a transformed component of the original transaction data.

- Important: The exact meaning of V1–V28 is not publicly available because these features are anonymized transformations of the original transaction data.
- 3. Amount :
Represents the monetary value of each credit card transaction.
Contains the transaction amount.
Can be useful for identifying unusual transaction patterns.
- 4. Class (Target Variable) :
The Class column represents the target variable used for binary classification.
0 : Legitimate Transaction
1 : Fraudulent Transaction
- The model attempts to predict whether a transaction belongs to Class 0 or Class 1.
##  4. Class Imbalance :
- One of the main challenges of this dataset is class imbalance.
The distribution of transactions is:
- Class :
0 (Legitimate) : 284,315 , 99.8273%
1 (Fraudulent) : 492 , 0.1727%
- This imbalance can cause Machine Learning models to favor the majority class.
For example, a model could achieve very high accuracy simply by predicting every transaction as legitimate.
To address this issue, techniques such as the following can be investigated:
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Undersampling
- Class Weight Adjustment
- Threshold Optimization
## 5. Machine Learning Algorithms
### The following algorithms can be implemented and compared:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
## 6. Evaluation Metrics
### Since the dataset is highly imbalanced, Accuracy alone is not sufficient.
###The following evaluation metrics will be used:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
## 7. Technologies & Libraries
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
## 8. Project Goals
### The main goals of this project are:
- Explore and analyze credit card transaction data.
- Understand class imbalance.
- Perform data preprocessing and feature analysis.
- Train multiple Machine Learning algorithms.
- Apply techniques to handle imbalanced datasets.
- Compare model performance.
- Identify the most suitable model for fraudulent transaction detection.

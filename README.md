# 🏦 Bank Customer Churn Prediction: Stay or Go?

This project implements a Machine Learning solution to predict which customers are likely to leave the bank (churn). By identifying at-risk customers based on their banking behavior, the institution can take proactive measures to improve retention and loyalty.

## 📋 Project Overview
Customer churn is a critical metric for banks as acquiring a new customer is 5-7 times more expensive than retaining an existing one. In this project, I focused on maximizing **Recall** for the churned class (Class 1) to ensure the bank captures as many potential "leavers" as possible.

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
* **Balancing:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Models Tested:** * Logistic Regression
    * Support Vector Machine (SVM)
    * Decision Tree
    * **Random Forest**
    * **XGBoost** (Final Choice)

## 📊 Workflow
1. **Data Cleaning:** Removed non-informative identifiers like `RowNumber`, `CustomerId`, and `Surname`.
2. **Preprocessing:** Applied One-Hot Encoding for categorical features (`Geography`, `Gender`) and `StandardScaler` for numerical features.
3. **Imbalance Handling:** Used SMOTE to balance the training set, increasing the churned class samples to match the majority class.
4. **Evaluation:** Compared models using Confusion Matrices and Classification Reports, focusing on the F1-Score/Recall trade-off.



[Image of machine learning workflow diagram]


## 📈 Performance Results
After testing multiple algorithms, the ensemble models provided the best balance between accuracy and the ability to detect churners:

| Model | Accuracy | Recall (Class 1) | F1-Score |
| :--- | :--- | :--- | :--- |
| **SVM** | **0.864** | 0.40 | 0.55 |
| **Random Forest** | 0.861 | 0.46 | **0.57** |
| **XGBoost** | 0.851 | **0.49** | **0.57** |
| **Logistic Regression** | 0.808 | 0.19 | 0.28 |
| **Decision Tree** | 0.787 | 0.53 | 0.50 |

## 🏆 Key Insights & Final Model
The **XGBoost** model was selected for final deployment.

* **Feature Importance:** **Age** was the most significant predictor of churn, followed by **Number of Products** and **Balance**.
* **Business Strategy:** Based on model insights, the bank should implement targeted loyalty programs for older demographics and simplify product bundles for multi-product holders.
* **Handling Imbalance:** Using SMOTE significantly improved the model's ability to recognize the minority class (churners) compared to baseline models.



## 📂 File Structure
* `customer_churn_analysis.ipynb`: Full EDA, preprocessing, and modeling pipeline.
* `churn_modelling.csv`: Raw banking dataset.
* `bank_churn_xgb_model.pkl`: The saved XGBoost model file.
* `standard_scaler.pkl`: Saved scaler for production data normalization.
* `README.md`: Project documentation.

## 🚀 Usage
You can load the trained model to predict churn probability for new customers:
```python
import joblib

# Load the model and scaler
model = joblib.load('bank_churn_xgb_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

# New customer data needs to be scaled before prediction
# prediction = model.predict(X_new_scaled)

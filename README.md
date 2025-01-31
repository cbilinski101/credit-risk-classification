# Loan Risk Prediction Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## **Overview of the Analysis**

The purpose of this analysis was to evaluate machine learning models for predicting loan risk classifications. Specifically, the goal was to determine whether an applicant's loan would be categorized as **low risk** or **high risk** based on financial data.

### **Financial Data and Prediction Goals**
The dataset contained financial information about loan applicants, including key variables related to credit history, income, debt-to-income ratios, and other financial indicators. The target variable was loan risk classification:

- **Low Risk (0)**
- **High Risk (1)**

To understand the distribution of the target variable, we analyzed its **value counts**, which showed a significant class imbalance—low-risk loans were far more prevalent than high-risk loans.

### **Machine Learning Process**
The analysis followed standard machine learning procedures:

1. **Data Preprocessing**  
   - Handled missing values and performed data cleaning in the `lending_data.csv` dataset.
   - Encoded categorical variables using one-hot encoding and standardized numerical features using `StandardScaler`.
   - Addressed class imbalance using resampling techniques such as SMOTE (Synthetic Minority Over-sampling Technique).

2. **Model Selection and Training**  
   - Used **Logistic Regression** as the primary classification model.
   - Implemented the model within the `credit_risk_classification.ipynb` Jupyter Notebook.
   - Other models may have been explored depending on performance.

3. **Model Evaluation**  
   - Measured model accuracy, precision, recall, and F1-score.
   - Focused on **how well the model identified high-risk loans**, as misclassifying them could have serious financial implications.

---

## **Results**

### **Machine Learning Model 1: Logistic Regression**

- **Accuracy:** 99%
- **Precision:**
  - **Low Risk:** 1.00  
  - **High Risk:** 0.84  
- **Recall:**
  - **Low Risk:** 0.99  
  - **High Risk:** 0.94  
- **F1-Score:**
  - **Low Risk:** 1.00  
  - **High Risk:** 0.89  

The model performs **exceptionally well on low-risk loans** with nearly perfect precision and recall. However, while recall for high-risk loans is relatively strong (0.94), precision is lower (0.84), meaning that some **high-risk predictions are false positives**.

---

## **Summary and Recommendation**

- **Best Performing Model:** The logistic regression model achieves **high overall accuracy (99%)**, making it an effective classifier for loan risk assessment.  
- **Model Strengths:** The model is **highly reliable for identifying low-risk loans**, with nearly perfect precision and recall.  
- **Model Weaknesses:** While recall for high-risk loans is strong, its **lower precision (0.84)** means that some low-risk loans may be incorrectly classified as high-risk.  

### **Recommendation**
The choice of the best model depends on business priorities:

- If the goal is to **minimize false negatives** (i.e., avoid misclassifying high-risk loans as low-risk), this model performs well because of its high recall for high-risk loans.
- If **false positives for high-risk loans** are a major concern (i.e., denying loans incorrectly), then additional tuning or alternative models may be needed.

Overall, the **logistic regression model is a strong candidate** but may need adjustments, such as hyperparameter tuning or balancing techniques, to improve high-risk loan classification further.

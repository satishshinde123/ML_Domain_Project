# Banking Customer Churn Prediction

## Project Overview

The **Banking Customer Churn Prediction** project is an end-to-end Machine Learning classification project designed to predict whether a banking customer is likely to churn.

The project uses customer demographic information, account details, financial attributes, transaction activity, digital banking usage, customer-service interactions, and banking product ownership to predict the target variable:

**`Churn` — Yes / No**

The project is structured as a standardized ML workflow covering data understanding, data cleaning, feature preparation, model development, evaluation, and business recommendations.

---

## Business Problem

Customer churn is an important business problem for retail banks because acquiring a new customer can be more expensive than retaining an existing customer.

The objective of this project is to build a Machine Learning model that can identify customers who are at higher risk of leaving the bank.

The prediction can help the bank:

- Identify high-risk customers.
- Prioritize customer-retention activities.
- Improve customer engagement.
- Support personalized retention campaigns.
- Improve digital banking adoption.
- Identify customers requiring proactive relationship-manager intervention.

---

## Machine Learning Problem

### Problem Type

**Supervised Learning — Binary Classification**

### Target Variable

```text
Churn
```

Target classes:

```text
Yes → Customer is likely to churn
No  → Customer is likely to remain
```

---

## Dataset

### Dataset Name

```text
banking_customer_churn_50000.csv
```

The generated dataset contains banking customer information suitable for an end-to-end ML practice project.

The dataset intentionally contains data-quality issues so that the complete preprocessing workflow can be demonstrated.

### Dataset Characteristics

- Approximately 50,000 original customer records.
- Additional duplicate records are included intentionally for data-cleaning practice.
- Numerical and categorical features.
- Missing values.
- Invalid/error values such as `?`, `@`, `#`, `%`.
- Outliers in selected financial and activity variables.
- Binary churn target.

---

## Important Features

The dataset contains features related to:

### Customer Demographics

- `Age`
- `Gender`
- `Education`
- `Marital_Status`
- `Employment_Status`
- `City`
- `Branch_Type`

### Banking Relationship

- `Account_Type`
- `Account_Tenure_Years`
- `Relationship_Manager`

### Digital Banking

- `Digital_Banking`
- `Mobile_Banking`
- `Internet_Banking`

### Financial Profile

- `Annual_Income`
- `Credit_Score`
- `Employment_Years`
- `Existing_Loans`
- `Loan_Amount`
- `Monthly_EMI`
- `Debt_to_Income`
- `Account_Balance`

### Transaction Activity

- `Avg_Monthly_Transactions`
- `Cash_Withdrawals`
- `Online_Transactions`
- `Failed_Transactions`
- `Last_Transaction_Days`

### Credit Card

- `Credit_Card`
- `Credit_Card_Spend`

### Customer Service

- `Customer_Service_Calls`
- `Complaints`
- `Branch_Visits`

### Banking Products

- `Fixed_Deposit`
- `Insurance`
- `Mutual_Fund`
- `Personal_Loan`
- `Home_Loan`

### Target

- `Churn`

---

## Project Structure

```text
Banking_Customer_Churn/
│
├── Banking_Customer_Churn_Standardized.ipynb
├── banking_customer_churn_50000.csv
├── README.md
│
├── banking_customer_churn_model.pkl
├── banking_churn_predictions.csv
│
└── Banking_Customer_Churn_Documents/
    ├── 2_Data_Understanding_Report_Banking.docx
    ├── 3_Feature_Engineering_Report_Banking.docx
    ├── 4_Model_Development_Report_Banking.docx
    ├── 5_Model_Evaluation_Report_Banking.docx
    └── 6_Insights_Business_Recommendations_Report_Banking.docx
```

---

# ML Workflow

```text
Business Problem
       ↓
Data Loading
       ↓
Data Understanding
       ↓
Duplicate Handling
       ↓
Error Value Handling
       ↓
Missing Value Handling
       ↓
Outlier Treatment
       ↓
EDA
       ↓
Feature / Target Split
       ↓
Categorical Encoding
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Multiple Model Training
       ↓
Model Comparison
       ↓
Best Model Selection
       ↓
Model Evaluation
       ↓
Feature Importance
       ↓
Prediction
       ↓
Model Persistence
       ↓
Real-Time Prediction
```

---

# Data Preprocessing

## 1. Duplicate Handling

Duplicate records are identified using:

```python
df.duplicated().sum()
```

Duplicate records are removed before model development.

---

## 2. Error Value Handling

The dataset contains error values:

```python
error_values = ['?', '@', '#', '%']
```

These values are replaced with `NaN` before imputation.

---

## 3. Missing Value Handling

### Categorical Variables

Missing categorical values are replaced with the mode:

```python
df[col] = df[col].fillna(df[col].mode()[0])
```

### Numerical Variables

Missing numerical values are replaced with the median:

```python
df[col] = df[col].fillna(df[col].median())
```

---

## 4. Outlier Handling

The project uses the **Interquartile Range (IQR)** method.

```text
IQR = Q3 - Q1

Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

Instead of deleting customer records, extreme values are capped at the IQR boundaries.

This preserves customer observations while reducing the effect of extreme values.

---

# Feature Preparation

## Customer ID

`Customer_ID` is removed from the feature set because it is an identifier and does not represent a meaningful predictive characteristic.

```python
X = df.drop(
    columns=['Customer_ID', 'Churn']
)
```

## Categorical Encoding

Categorical features are converted into numerical values using `LabelEncoder` in the standardized notebook.

## Target Encoding

The target:

```text
Churn
```

is encoded into binary numerical values using `LabelEncoder`.

---

# Train-Test Split

The project uses an **80:20 stratified train-test split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification helps maintain a similar churn-class distribution in the training and testing datasets.

---

# Feature Scaling

`StandardScaler` is used to standardize the numerical feature matrix.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training dataset and then applied to the test dataset to avoid data leakage.

---

# Machine Learning Models

The project compares multiple classification algorithms.

## 1. Logistic Regression

Used as an interpretable baseline classification model.

## 2. Decision Tree

Captures nonlinear relationships using tree-based decision rules.

## 3. Random Forest

An ensemble of decision trees that can capture nonlinear relationships and provide feature importance.

## 4. Gradient Boosting

Builds an ensemble sequentially and can model complex nonlinear relationships.

## 5. K-Nearest Neighbors

Provides a distance-based classification benchmark.

---

# Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Classification Report
- Confusion Matrix

## Why F1-Score?

The banking churn target is imbalanced, so accuracy alone can be misleading.

F1-score balances:

```text
Precision + Recall
```

For a churn prediction system, **Recall** is particularly important because missing a customer who is actually going to churn can reduce the effectiveness of retention campaigns.

---

# Model Selection

The standardized notebook compares the models and selects the best model based on the highest **F1-score**.

The selected model is then trained and evaluated on the test dataset.

---

# Confusion Matrix

The confusion matrix contains:

```text
True Negative
False Positive
False Negative
True Positive
```

For churn prediction:

- **True Positive:** Correctly identifies a customer who churns.
- **False Positive:** Predicts churn for a customer who does not churn.
- **False Negative:** Fails to identify a customer who actually churns.
- **True Negative:** Correctly identifies a customer who does not churn.

---

# Feature Importance

For tree-based models, feature importance is obtained from:

```python
model.feature_importances_
```

For Logistic Regression, absolute coefficient values are used.

Feature importance helps identify the customer characteristics that contribute most to the model's predictions.

---

# Model Persistence

The trained model and preprocessing objects are saved using `joblib`.

```text
banking_customer_churn_model.pkl
```

The saved package contains:

- Trained model
- StandardScaler
- Label encoders
- Target encoder
- Feature column order
- Target variable

This allows the same preprocessing and model to be reused during inference.

---

# Prediction Output

Predictions can be saved as:

```text
banking_churn_predictions.csv
```

The prediction output contains customer information together with:

```text
Actual_Churn
Predicted_Churn
Churn_Probability
```

---

# Real-Time Prediction

The notebook contains a reusable function:

```python
predict_churn(new_data)
```

Example:

```python
result = predict_churn(sample_customer)
```

The function returns:

```text
Predicted_Churn
Churn_Probability
```

This structure can later be integrated with:

- REST API
- Web application
- Banking dashboard
- Batch prediction pipeline
- Customer-retention system

---

# Business Use Case

The model can be used as a **customer-retention decision-support system**.

A possible business workflow is:

```text
Customer Data
     ↓
ML Churn Prediction
     ↓
Churn Probability
     ↓
Risk Segmentation
     ↓
Retention Strategy
```

For example:

| Risk Level | Example Action |
|---|---|
| Low | Normal customer engagement |
| Medium | Personalized communication |
| High | Priority relationship-manager intervention |

The exact probability thresholds should be optimized using business cost/benefit analysis before production deployment.

---

# Business Recommendations

Potential retention actions include:

### 1. Reduce Customer Inactivity

Monitor customers with long periods since their last transaction.

### 2. Increase Digital Banking Adoption

Encourage customers to use mobile and internet banking.

### 3. Proactive Complaint Management

Prioritize customers with repeated complaints or service interactions.

### 4. Personalized Retention Offers

Use churn probability together with customer value to design targeted offers.

### 5. Relationship Management

Prioritize high-risk customers for relationship-manager engagement.

### 6. Product Engagement

Use relevant banking products and services to strengthen the customer relationship.

---

# Project Documentation

The project includes five supporting documents:

1. **Data Understanding Report**
2. **Feature Engineering Report**
3. **Model Development Report**
4. **Model Evaluation Report**
5. **Insights & Business Recommendations Report**

These documents provide detailed project documentation from data understanding through business recommendations.

---

# Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Jupyter Notebook / Google Colab
```

---

# Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

# How to Run

## Option 1 — Jupyter Notebook

Open:

```text
Banking_Customer_Churn_Standardized.ipynb
```

Make sure the dataset is in the same directory:

```text
banking_customer_churn_50000.csv
```

Run the notebook cells sequentially.

## Option 2 — Google Colab

Upload:

```text
Banking_Customer_Churn_Standardized.ipynb
banking_customer_churn_50000.csv
```

Then execute the notebook from the first cell to the last cell.

---

# Important ML Engineering Considerations

This project is a strong end-to-end ML practice project, but the current implementation should be considered a **baseline model**, not a production banking decision system.

Before production deployment, consider:

- One-Hot Encoding for nominal categorical variables.
- A unified `Pipeline` / `ColumnTransformer`.
- Cross-validation-based hyperparameter tuning.
- Class imbalance techniques such as class weighting or SMOTE where appropriate.
- Probability calibration.
- Business-specific prediction threshold optimization.
- Model explainability.
- Data drift monitoring.
- Model performance monitoring.
- Periodic model retraining.
- Security and privacy controls for banking data.
- Validation on genuinely future customer data.

---

# Future Enhancements

Possible next versions can include:

```text
Hyperparameter Tuning
        ↓
SMOTE / Class Weight Optimization
        ↓
Threshold Optimization
        ↓
Probability Calibration
        ↓
SHAP Explainability
        ↓
MLflow Experiment Tracking
        ↓
REST API
        ↓
Docker
        ↓
Cloud Deployment
        ↓
Model Monitoring
```

---

# Project Outcome

The project demonstrates an end-to-end Machine Learning workflow for banking customer churn prediction:

**Data → Cleaning → Feature Engineering → Modeling → Evaluation → Prediction → Business Recommendation**

The final system provides a reusable baseline for identifying customers at risk of churn and supporting data-driven customer-retention strategies.

---

## Author

**Banking Customer Churn Prediction — Machine Learning Project**


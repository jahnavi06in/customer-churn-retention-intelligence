# Customer Churn & Retention Intelligence

A production-style Streamlit analytics project that combines:

- Customer segmentation
- Segmentation validation
- Churn prediction
- Customer risk scoring
- Retention prioritization
- Business-focused customer exploration

## Business Question

> Which customers are most likely to churn, which customer groups are exposed to the highest risk, and who should the retention team prioritize?

## Architecture

```text
Churn_Modelling.csv
        |
        v
Data Validation
        |
        +----------------------+
        |                      |
        v                      v
Customer Segmentation     Churn Prediction
        |                      |
        v                      v
Segment Profiles          Risk Probability
        |                      |
        +----------+-----------+
                   |
                   v
          Retention Priority
                   |
                   v
        Streamlit Decision App
```

## Analytical Approach

### 1. Segmentation

K-Means is evaluated across k=2 to k=6 using silhouette score.

The segmentation features include:

- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- EstimatedSalary
- Geography
- Gender

Numeric variables are standardized and categorical variables are one-hot encoded.

The app exposes the silhouette scores rather than presenting the resulting clusters as objectively "true" customer personas.

### 2. Churn Prediction

Three models are trained:

- Logistic Regression
- Random Forest
- Gradient Boosting

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

The model with the highest ROC-AUC is used for customer risk scoring.

### 3. Risk Scoring

Every customer receives:

- Churn probability
- Risk level
- Retention priority
- Recommended action

Risk thresholds:

- Low: <30%
- Medium: 30–60%
- High: >60%

These thresholds are business rules and should be recalibrated using historical retention campaign outcomes in a real production setting.

## Dashboard

### Executive Overview
High-level churn KPIs, geographic churn and risk distribution.

### Customer Segmentation
Silhouette validation, segment profiles and customer segment visualization.

### Churn Intelligence
Model comparison and predicted risk by geography.

### Retention Strategy
High-priority customers and recommended retention actions.

### Customer Explorer
Filterable customer-level risk and segment information.

## Run Locally

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Add the dataset

Put:

```text
data/Churn_Modelling.csv
```

### 4. Run

```powershell
streamlit run app.py
```

## Project Structure

```text
Customer_Churn_Retention_Intelligence/
│
├── app.py
├── app/
│   ├── __init__.py
│   └── dashboard.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── segmentation.py
│   └── churn_model.py
│
├── data/
│   └── README.md
│
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Important Technical Notes

The original version of this project selected K-Means solely by the maximum silhouette score and produced weak clusters. This version makes the evaluation visible and limits the search to k=2..6.

The dataset's binary variables are not blindly standardized into the clustering feature matrix. Geography and Gender are encoded explicitly, while HasCrCard and IsActiveMember are retained as useful predictive features for the churn model.

Cluster labels are analytical groupings, not causal explanations.

The churn model identifies statistical risk; it does not prove that a customer will leave.

## Resume-Ready Project Description

**Customer Churn & Retention Intelligence | Python, Scikit-learn, Streamlit, Plotly**

Built an end-to-end customer retention analytics application combining validated K-Means segmentation with supervised churn prediction. Compared clustering structures using silhouette analysis, trained Logistic Regression, Random Forest and Gradient Boosting models, and converted customer-level churn probabilities into actionable retention priorities and recommended interventions.

## GitHub

Before pushing:

```powershell
git init
git add .
git commit -m "Build customer churn retention intelligence dashboard"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

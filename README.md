# Customer Churn & Retention Intelligence

> An end-to-end customer analytics and machine learning application that identifies churn risk, segments customers, and converts predictive insights into actionable retention priorities.

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-red?logo=streamlit)](https://customer-churn-retention-intelligence-rtmjld9xsqh5bxxr5bdqqj.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-black?logo=github)](https://github.com/Jahnavi06in/customer-churn-retention-intelligence)

---

## 🚀 Live Dashboard

### [Open the Customer Churn & Retention Intelligence Dashboard](https://customer-churn-retention-intelligence-rtmjld9xsqh5bxxr5bdqqj.streamlit.app/)

An interactive Streamlit decision-support application covering:

- 📊 Executive Overview
- 👥 Customer Segmentation
- 🤖 Churn Intelligence
- ⚠️ Customer Risk Analysis
- 🎯 Retention Strategy
- 🔎 Customer Explorer

---

## 🎯 Business Problem

Customer churn is not only a prediction problem. Businesses also need to understand:

- Which customers are most likely to churn?
- Which customer groups are exposed to higher risk?
- How strong is the predictive model?
- Which customers should the retention team prioritize?
- How can analytical predictions be translated into business actions?

This project addresses these questions through a combination of **customer segmentation, supervised machine learning, risk scoring, and retention prioritization**.

---

## 🧩 Solution Architecture

```text
Churn_Modelling.csv
        │
        ▼
Data Validation & Exploration
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
Customer Segmentation    Churn Prediction
        │                      │
        ▼                      ▼
Segment Profiles         Model Evaluation
                               │
                               ▼
                        Churn Probability
                               │
                               ▼
                         Risk Scoring
                               │
                               ▼
                    Retention Prioritization
                               │
                               ▼
                    Streamlit Decision App

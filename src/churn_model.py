import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
CATEGORICAL = ["Geography", "Gender"]

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

def train_models(df, random_state=42):
    X = df[NUMERIC + CATEGORICAL]
    y = df["Exited"]

    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }

    fitted = {}
    metrics = []

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X)[:, 1]

        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y, proba),
        })
        fitted[name] = pipe

    return fitted, pd.DataFrame(metrics)

def add_risk_scores(df, model):
    result = df.copy()
    X = result[NUMERIC + CATEGORICAL]
    result["Churn_Probability"] = model.predict_proba(X)[:, 1]

    result["Risk_Level"] = pd.cut(
        result["Churn_Probability"],
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    result["Retention_Priority"] = (
        result["Churn_Probability"]
        * (1 + result["Balance"].fillna(0) / max(result["Balance"].max(), 1))
    )

    result["Recommended_Action"] = result.apply(recommend_action, axis=1)
    return result

def recommend_action(row):
    if row["Risk_Level"] == "High" and row["Balance"] > 100000:
        return "Priority retention outreach"
    if row["Risk_Level"] == "High":
        return "Retention offer + relationship review"
    if row["Risk_Level"] == "Medium" and row["IsActiveMember"] == 0:
        return "Re-engagement campaign"
    if row["Risk_Level"] == "Medium":
        return "Monitor + personalized engagement"
    return "Maintain relationship"

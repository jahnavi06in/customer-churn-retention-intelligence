import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score

NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary"
]
CATEGORICAL = ["Geography", "Gender"]

def prepare_features(df):
    X = df[NUMERIC + CATEGORICAL].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, preprocessor

def evaluate_k(df, k_values=range(2, 7), random_state=42):
    X, _ = prepare_features(df)
    rows = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette_score": score})

    return pd.DataFrame(rows)

def fit_segmentation(df, k=4, random_state=42):
    X, preprocessor = prepare_features(df)

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=20,
    )
    labels = model.fit_predict(X)

    result = df.copy()
    result["Segment"] = labels

    profile = (
        result.groupby("Segment")
        .agg(
            Customers=("CustomerId", "count"),
            Churn_Rate=("Exited", "mean"),
            Avg_Age=("Age", "mean"),
            Avg_Balance=("Balance", "mean"),
            Avg_CreditScore=("CreditScore", "mean"),
            Avg_Products=("NumOfProducts", "mean"),
            Active_Rate=("IsActiveMember", "mean"),
        )
        .reset_index()
    )

    # Deterministic business labels based on observed profiles.
    # Labels are deliberately descriptive rather than pretending clusters
    # are naturally occurring "true" personas.
    profile["Churn_Rate"] = profile["Churn_Rate"] * 100
    profile["Active_Rate"] = profile["Active_Rate"] * 100

    ordered = profile.sort_values(
        ["Churn_Rate", "Active_Rate"], ascending=[False, True]
    )["Segment"].tolist()

    names = [
        "High-Risk Customers",
        "Growth Opportunity",
        "Stable Customers",
        "Core Customers",
        "Emerging Customers",
    ]

    name_map = {
        seg: names[i] if i < len(names) else f"Segment {i+1}"
        for i, seg in enumerate(ordered)
    }

    result["Persona"] = result["Segment"].map(name_map)
    profile["Persona"] = profile["Segment"].map(name_map)

    return result, profile, model, preprocessor

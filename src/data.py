from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited"
]

def load_data(path="data/Churn_Modelling.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put Churn_Modelling.csv inside the data/ folder."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["Exited"] = pd.to_numeric(df["Exited"], errors="coerce")
    df = df.dropna(subset=["Exited"])
    df["Exited"] = df["Exited"].astype(int)

    for col in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)

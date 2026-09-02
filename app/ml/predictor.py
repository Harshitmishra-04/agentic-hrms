import pandas as pd
import numpy as np
from app.ml.model_loader import load_pipeline


def predict_attrition_probabilities(employee_df: pd.DataFrame) -> pd.Series:
    """
    Compute attrition probabilities for a full employee DataFrame while preserving
    the exact feature schema the model was trained on.
    """
    pipe = load_pipeline()
    df = employee_df.copy()

    df["income_per_year_at_company"] = (df["MonthlyIncome"] * 12) / (df["YearsAtCompany"] + 1.0)
    df["years_since_last_promotion_gap"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["overall_satisfaction_composite"] = (
        df["EnvironmentSatisfaction"]
        + df["JobSatisfaction"]
        + df["RelationshipSatisfaction"]
        + df["WorkLifeBalance"]
    ) / 4.0
    df["experience_ratio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1.0)

    cols_to_drop = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber", "Attrition"]
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors="ignore")

    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    model_features = pipe.named_steps["scaler"].feature_names_in_
    X_final = pd.DataFrame(0.0, index=df_encoded.index, columns=model_features, dtype=float)
    for col in model_features:
        if col in df_encoded.columns:
            X_final[col] = df_encoded[col].astype(float).to_numpy()

    probs = pipe.predict_proba(X_final)[:, 1]
    return pd.Series(probs, index=df_encoded.index, name="attrition_probability")


def predict_attrition_probability(employee_data: dict) -> float:
    """
    Computes attrition probability for a single employee record.
    Accepts raw unscaled features as a dictionary.
    """
    if isinstance(employee_data, pd.DataFrame):
        return float(predict_attrition_probabilities(employee_data).iloc[0])
    if isinstance(employee_data, pd.Series):
        employee_data = employee_data.to_dict()
    return float(predict_attrition_probabilities(pd.DataFrame([employee_data])).iloc[0])

def get_risk_bucket(probability: float) -> str:
    """
    Returns the risk bucket for a given attrition probability.
    """
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"

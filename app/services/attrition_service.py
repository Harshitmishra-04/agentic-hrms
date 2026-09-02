import os
import pandas as pd
import numpy as np
from app.ml.model_loader import load_metadata
from app.ml.predictor import predict_attrition_probability, get_risk_bucket
from app.utils.prediction_logger import log_prediction

INTELLIGENCE_PATH = os.path.join("data", "processed", "employee_intelligence.csv")

def predict_single_employee(employee_data: dict) -> dict:
    """
    Computes attrition prediction and risk level for a single employee record.
    """
    prob = predict_attrition_probability(employee_data)
    risk = get_risk_bucket(prob)
    metadata = load_metadata() or {}
    model_version = str(metadata.get("version", "unknown"))
    employee_id = employee_data.get("EmployeeNumber", employee_data.get("employee_id"))
    log_prediction(
        employee_id=employee_id,
        model_version=model_version,
        probability=prob,
        risk_level=risk,
    )
    return {
        "attrition_probability": prob,
        "risk_bucket": risk
    }

def get_overall_attrition_summary() -> dict:
    """
    Returns overall statistics on attrition risk across the workforce.
    """
    if not os.path.exists(INTELLIGENCE_PATH):
        return {"average_probability": 0.0, "risk_counts": {}}
    
    df = pd.read_csv(INTELLIGENCE_PATH)
    avg_prob = float(df["attrition_probability"].mean())
    risk_counts = df["risk_bucket"].value_counts().to_dict()
    
    return {
        "average_probability": avg_prob,
        "risk_counts": risk_counts,
        "total_employees": int(df.shape[0])
    }

def get_attrition_by_department() -> list:
    """
    Returns attrition statistics aggregated by department.
    """
    if not os.path.exists(INTELLIGENCE_PATH):
        return []
    
    df = pd.read_csv(INTELLIGENCE_PATH)
    
    dept_stats = df.groupby("Department").agg(
        avg_probability=("attrition_probability", "mean"),
        total_employees=("employee_id", "count")
    ).reset_index()
    
    result = []
    for idx, row in dept_stats.iterrows():
        dept_name = row["Department"]
        df_dept = df[df["Department"] == dept_name]
        risk_dist = df_dept["risk_bucket"].value_counts().to_dict()
        
        result.append({
            "department": dept_name,
            "average_probability": float(row["avg_probability"]),
            "total_employees": int(row["total_employees"]),
            "risk_distribution": risk_dist
        })
        
    return result

def get_employee_intelligence(employee_id: int) -> dict:
    """
    Retrieves the master analytical record for a single employee.
    """
    if not os.path.exists(INTELLIGENCE_PATH):
        return None
        
    df = pd.read_csv(INTELLIGENCE_PATH)
    df_emp = df[df["employee_id"] == employee_id]
    
    if df_emp.empty:
        return None
        
    # Return as dict, converting NaN to None for JSON serializability
    record = df_emp.iloc[0].to_dict()
    for k, v in record.items():
        if pd.isna(v):
            record[k] = None
        elif isinstance(v, (np.integer, np.floating)):
            record[k] = v.item()
            
    return record

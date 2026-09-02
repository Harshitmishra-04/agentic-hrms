import os
import pandas as pd

ENGAGEMENT_PATH = os.path.join("data", "processed", "engagement_data.csv")
RAW_HR_PATH = os.path.join("data", "raw", "hr_performance_engagement.csv")

def get_overall_engagement_summary() -> dict:
    """
    Returns overall average engagement, satisfaction, and work-life balance scores.
    """
    if not os.path.exists(ENGAGEMENT_PATH):
        return {}
    
    df = pd.read_csv(ENGAGEMENT_PATH)
    return {
        "avg_engagement_score": float(df["Engagement Score"].mean()),
        "avg_satisfaction_score": float(df["Satisfaction Score"].mean()),
        "avg_work_life_balance_score": float(df["Work-Life Balance Score"].mean()),
        "total_records": int(df.shape[0])
    }

def get_engagement_by_department() -> list:
    """
    Groups engagement scores by employee department.
    """
    if not os.path.exists(ENGAGEMENT_PATH) or not os.path.exists(RAW_HR_PATH):
        return []
    
    df_eng = pd.read_csv(ENGAGEMENT_PATH)
    df_raw = pd.read_csv(RAW_HR_PATH)
    
    # Map department info using Employee ID / ID
    df_dept_map = df_raw[["Employee ID", "DepartmentType"]].drop_duplicates()
    df_joined = df_eng.merge(df_dept_map, on="Employee ID", how="inner")
    
    dept_stats = df_joined.groupby("DepartmentType").agg(
        avg_engagement_score=("Engagement Score", "mean"),
        avg_satisfaction_score=("Satisfaction Score", "mean"),
        avg_work_life_balance_score=("Work-Life Balance Score", "mean"),
        total_employees=("Employee ID", "count")
    ).reset_index()
    
    result = []
    for idx, row in dept_stats.iterrows():
        result.append({
            "department": row["DepartmentType"],
            "avg_engagement_score": float(row["avg_engagement_score"]),
            "avg_satisfaction_score": float(row["avg_satisfaction_score"]),
            "avg_work_life_balance_score": float(row["avg_work_life_balance_score"]),
            "total_employees": int(row["total_employees"])
        })
        
    return result

def get_lowest_engagement_records(limit: int = 20) -> list:
    """
    Returns a list of records with the lowest engagement scores to inform HR.
    """
    if not os.path.exists(ENGAGEMENT_PATH) or not os.path.exists(RAW_HR_PATH):
        return []
        
    df_eng = pd.read_csv(ENGAGEMENT_PATH)
    df_raw = pd.read_csv(RAW_HR_PATH)
    
    df_dept_map = df_raw[["Employee ID", "DepartmentType"]].drop_duplicates()
    df_joined = df_eng.merge(df_dept_map, on="Employee ID", how="inner")
    
    # Sort by engagement score ascending
    df_low = df_joined.sort_values(by="Engagement Score", ascending=True).head(limit)
    
    result = []
    for idx, row in df_low.iterrows():
        result.append({
            "employee_id": int(row["Employee ID"]),
            "department": row["DepartmentType"],
            "engagement_score": float(row["Engagement Score"]),
            "satisfaction_score": float(row["Satisfaction Score"]),
            "work_life_balance_score": float(row["Work-Life Balance Score"])
        })
        
    return result

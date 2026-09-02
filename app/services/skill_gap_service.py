import os
import pandas as pd

ORG_GAP_PATH = os.path.join("data", "processed", "org_skill_gaps.csv")
EMPLOYEE_GAP_PATH = os.path.join("data", "processed", "employee_skill_gaps.csv")

def get_org_skill_gaps_summary() -> dict:
    """
    Returns counts of HIGH, MEDIUM, and LOW severity gaps in the organization.
    Supports both legacy `skill` and current `missing_skill` schema names.
    """
    if not os.path.exists(ORG_GAP_PATH):
        return {}

    df = pd.read_csv(ORG_GAP_PATH)
    if "severity" not in df.columns:
        return {
            "total_unique_missing_skills": 0,
            "severity_distribution": {},
        }

    severity_counts = df["severity"].value_counts().to_dict()
    return {
        "total_unique_missing_skills": int(df.shape[0]),
        "severity_distribution": severity_counts,
    }

def get_org_avg_skill_gap_count() -> float:
    """
    Returns the average skill gap count per employee across the organization.
    """
    if not os.path.exists(EMPLOYEE_GAP_PATH):
        return 0.0
        
    df = pd.read_csv(EMPLOYEE_GAP_PATH)
    gap_counts = df.groupby("employee_id").size()
    return float(gap_counts.mean())

def get_top_missing_skills(limit: int = 15) -> list:
    """
    Returns the top organizational missing skills sorted by frequency.
    Supports both legacy `skill` and current `missing_skill` schema names.
    """
    if not os.path.exists(ORG_GAP_PATH):
        return []

    df = pd.read_csv(ORG_GAP_PATH)
    if "missing_count" not in df.columns:
        return []

    label_col = "missing_skill" if "missing_skill" in df.columns else "skill"
    df_sorted = df.sort_values(by="missing_count", ascending=False).head(limit)

    result = []
    for _, row in df_sorted.iterrows():
        result.append({
            "skill": row[label_col],
            "missing_count": int(row["missing_count"]),
            "avg_importance_score": float(row["avg_importance_score"]),
            "severity": row["severity"],
        })

    return result

def get_employee_skill_gaps(employee_id: int) -> list:
    """
    Returns the specific skill gaps and importance scores for a single employee.
    """
    if not os.path.exists(EMPLOYEE_GAP_PATH):
        return []
        
    df = pd.read_csv(EMPLOYEE_GAP_PATH)
    df_emp = df[df["employee_id"] == employee_id]
    
    result = []
    for idx, row in df_emp.iterrows():
        result.append({
            "skill": row["missing_skill"],
            "importance_score": float(row["importance_score"])
        })
        
    return result

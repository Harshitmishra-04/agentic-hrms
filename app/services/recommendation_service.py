import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
COURSES_PATH = os.path.join(DATA_PROCESSED, "courses.csv")
RECS_PIVOTED_PATH = os.path.join(DATA_PROCESSED, "employee_recommendations_pivoted_v3.csv")
RECS_LONG_PATH = os.path.join(DATA_PROCESSED, "employee_course_recommendations_v3.csv")

def get_course_catalog() -> list:
    """
    Returns the complete list of training courses.
    """
    if not os.path.exists(COURSES_PATH):
        return []
        
    df = pd.read_csv(COURSES_PATH)
    result = []
    for idx, row in df.iterrows():
        result.append({
            "course_title": row["course_title"],
            "target_skill": row["target_skill"],
            "difficulty": row["difficulty"],
            "duration_days": int(row["duration_days"])
        })
    return result

def get_recommendations_summary() -> dict:
    """
    Returns counts and distributions of course recommendations.
    """
    catalog = get_course_catalog()
    if not os.path.exists(RECS_LONG_PATH):
        return {
            "total_recommendations": 0,
            "catalog": catalog,
            "summary": {"course_distribution": {}},
        }

    df = pd.read_csv(RECS_LONG_PATH)
    course_counts = df["recommended_course"].value_counts().to_dict()
    
    return {
        "total_recommendations": int(df.shape[0]),
        "catalog": catalog,
        "summary": {
            "course_distribution": course_counts
        }
    }

def get_employee_recommendations(employee_id: int) -> dict:
    """
    Returns the top 3 course recommendations for a single employee.
    """
    if not os.path.exists(RECS_PIVOTED_PATH):
        return {}
        
    df = pd.read_csv(RECS_PIVOTED_PATH)
    df_emp = df[df["employee_id"] == employee_id]
    
    if df_emp.empty:
        return {}
        
    row = df_emp.iloc[0]
    return {
        "recommended_course_1": row.get("recommended_course_1", None),
        "recommended_course_2": row.get("recommended_course_2", None),
        "recommended_course_3": row.get("recommended_course_3", None)
    }

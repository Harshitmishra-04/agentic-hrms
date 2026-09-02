import pandas as pd

from app.services.skill_gap_service import get_employee_skill_gaps


def test_skill_gap_calculation_matches_expected_output():
    employee_id = 1
    employee_skills = pd.read_csv("data/processed/employee_skills.csv")
    role_skills = pd.read_csv("data/processed/role_skills.csv")
    employee_gaps = pd.read_csv("data/processed/employee_skill_gaps.csv")

    possessed = set(
        employee_skills.loc[employee_skills["employee_id"] == employee_id, "current_skill"]
    )
    onet_code = employee_gaps.loc[
        employee_gaps["employee_id"] == employee_id, "onet_soc_code"
    ].iloc[0]
    required = set(
        role_skills.loc[role_skills["O*NET-SOC Code"] == onet_code, "Skill Name"]
    )
    expected_missing = required - possessed

    service_missing = {row["skill"] for row in get_employee_skill_gaps(employee_id)}

    assert service_missing == expected_missing
    assert len(service_missing) > 0
    assert possessed.isdisjoint(service_missing)

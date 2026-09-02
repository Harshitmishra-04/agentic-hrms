from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from app.services import attrition_service, engagement_service, recommendation_service, skill_gap_service, rag_service, agentic_service
from app.utils.logging_config import get_logger
from app.validation.schemas import (
    AttritionPredictRequest,
    DashboardQuery,
    EmployeeLookupRequest,
    SkillGapsQuery,
    RAGQueryRequest,
    AgentWorkflowRequest,
)

router = APIRouter()


@router.post("/predict/attrition")
def predict_attrition(employee_data: AttritionPredictRequest):
    """
    Predicts attrition probability and risk bucket for a raw employee record.
    Invalid payloads return 400 before the model is invoked.
    """
    logger = get_logger()
    logger.info(
        "Attrition prediction requested for employee_id=%s",
        employee_data.EmployeeNumber,
    )
    try:
        result = attrition_service.predict_single_employee(
            employee_data.model_dump(by_alias=True, exclude_none=True)
        )
        return result
    except Exception as e:
        logger.exception("Attrition prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dashboard/summary")
def get_dashboard_summary(_query: Annotated[DashboardQuery, Depends()]):
    """
    Retrieves high-level summary statistics across attrition, engagement, and skill gaps.
    """
    attr_summary = attrition_service.get_overall_attrition_summary()
    eng_summary = engagement_service.get_overall_engagement_summary()
    skills_summary = skill_gap_service.get_org_skill_gaps_summary()
    avg_skill_gap_count = skill_gap_service.get_org_avg_skill_gap_count()
    recs_summary = recommendation_service.get_recommendations_summary()

    return {
        "attrition": attr_summary,
        "engagement": eng_summary,
        "skills": {**skills_summary, "avg_skill_gap_count": avg_skill_gap_count},
        "recommendations": recs_summary,
    }


@router.get("/dashboard/attrition-by-department")
def get_department_attrition(_query: Annotated[DashboardQuery, Depends()]):
    """
    Retrieves department-level attrition statistics and risk distributions.
    """
    return attrition_service.get_attrition_by_department()


@router.get("/dashboard/skill-gaps")
def get_skill_gaps(query: Annotated[SkillGapsQuery, Depends()]):
    """
    Retrieves organization-wide skill gap summary and the top missing skills.
    """
    summary = skill_gap_service.get_org_skill_gaps_summary()
    top_skills = skill_gap_service.get_top_missing_skills(limit=query.limit)
    return {
        "summary": summary,
        "top_missing_skills": top_skills,
    }


@router.get("/dashboard/recommendations")
def get_recommendations(_query: Annotated[DashboardQuery, Depends()]):
    """
    Retrieves the course catalog and distribution counts for recommended courses.
    """
    catalog = recommendation_service.get_course_catalog()
    summary = recommendation_service.get_recommendations_summary()
    return {
        "catalog": catalog,
        "summary": summary,
    }


@router.get("/employees/{employee_id}")
def get_employee_record(employee_id: Annotated[int, Path(gt=0)]):
    """
    Retrieves the master analytical record for a specific employee ID.
    """
    params = EmployeeLookupRequest(employee_id=employee_id)
    record = attrition_service.get_employee_intelligence(params.employee_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Employee ID {employee_id} not found.")
    return record


@router.post("/rag/ask")
def ask_rag_question(request: RAGQueryRequest):
    """
    Answers HR policy questions using hybrid retrieval (BM25 + vector) + reranking.
    """
    logger = get_logger()
    logger.info("RAG query received: %s", request.query)
    
    try:
        result = rag_service.ask_question(request.query, top_k=request.top_k)
        return result
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/workforce-check")
def run_workforce_check(request: AgentWorkflowRequest):
    """
    Runs a 5-step tool sequence for workforce intelligence:
    1. get_employee_profile
    2. get_skills
    3. calculate_skill_gap
    4. recommend_courses
    5. generate_learning_plan
    
    Returns a structured trace with each step's tool name, inputs, and output.
    Respects ToolAuthorizer permissions - blocked steps are reported clearly.
    """
    try:
        return agentic_service.run_workforce_check(request.employee_id, request.user_role)
    except Exception as e:
        logger.exception("Workforce check failed")
        raise HTTPException(status_code=500, detail=str(e))

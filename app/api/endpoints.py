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
        rag = rag_service.get_rag_service()
        result = rag.ask(request.query, top_k=request.top_k)
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
    logger = get_logger()
    logger.info("Workforce check requested for employee_id=%s by user_role=%s", request.employee_id, request.user_role)
    
    try:
        # Import tools directly to run them sequentially
        from app.services.agentic_service import (
            get_employee_profile,
            get_skills,
            calculate_skill_gap,
            recommend_courses,
            generate_learning_plan,
            ToolAuthorizer
        )
        
        trace = []
        employee_id = request.employee_id
        user_role = request.user_role
        
        # Step 1: get_employee_profile
        step_1 = {
            "step": 1,
            "tool": "get_employee_profile",
            "inputs": {"employee_id": employee_id, "user_role": user_role},
            "status": "pending"
        }
        
        if ToolAuthorizer.authorize("get_employee_profile", user_role):
            try:
                result = get_employee_profile.invoke({"employee_id": employee_id, "user_role": user_role})
                step_1["output"] = result
                step_1["status"] = "success"
            except Exception as e:
                step_1["status"] = "error"
                step_1["error"] = str(e)
        else:
            step_1["status"] = "blocked"
            step_1["blocked_reason"] = f"User role '{user_role}' is not authorized to use tool 'get_employee_profile'"
        
        trace.append(step_1)
        
        # Step 2: get_skills
        step_2 = {
            "step": 2,
            "tool": "get_skills",
            "inputs": {"employee_id": employee_id, "user_role": user_role},
            "status": "pending"
        }
        
        if ToolAuthorizer.authorize("get_skills", user_role):
            try:
                result = get_skills.invoke({"employee_id": employee_id, "user_role": user_role})
                step_2["output"] = result
                step_2["status"] = "success"
            except Exception as e:
                step_2["status"] = "error"
                step_2["error"] = str(e)
        else:
            step_2["status"] = "blocked"
            step_2["blocked_reason"] = f"User role '{user_role}' is not authorized to use tool 'get_skills'"
        
        trace.append(step_2)
        
        # Step 3: calculate_skill_gap
        step_3 = {
            "step": 3,
            "tool": "calculate_skill_gap",
            "inputs": {"employee_id": employee_id, "user_role": user_role},
            "status": "pending"
        }
        
        if ToolAuthorizer.authorize("calculate_skill_gap", user_role):
            try:
                result = calculate_skill_gap.invoke({"employee_id": employee_id, "user_role": user_role})
                step_3["output"] = result
                step_3["status"] = "success"
            except Exception as e:
                step_3["status"] = "error"
                step_3["error"] = str(e)
        else:
            step_3["status"] = "blocked"
            step_3["blocked_reason"] = f"User role '{user_role}' is not authorized to use tool 'calculate_skill_gap'"
        
        trace.append(step_3)
        
        # Step 4: recommend_courses
        step_4 = {
            "step": 4,
            "tool": "recommend_courses",
            "inputs": {"employee_id": employee_id, "user_role": user_role},
            "status": "pending"
        }
        
        if ToolAuthorizer.authorize("recommend_courses", user_role):
            try:
                result = recommend_courses.invoke({"employee_id": employee_id, "user_role": user_role})
                step_4["output"] = result
                step_4["status"] = "success"
            except Exception as e:
                step_4["status"] = "error"
                step_4["error"] = str(e)
        else:
            step_4["status"] = "blocked"
            step_4["blocked_reason"] = f"User role '{user_role}' is not authorized to use tool 'recommend_courses'"
        
        trace.append(step_4)
        
        # Step 5: generate_learning_plan
        step_5 = {
            "step": 5,
            "tool": "generate_learning_plan",
            "inputs": {"employee_id": employee_id, "user_role": user_role},
            "status": "pending"
        }
        
        if ToolAuthorizer.authorize("generate_learning_plan", user_role):
            try:
                result = generate_learning_plan.invoke({"employee_id": employee_id, "user_role": user_role})
                step_5["output"] = result
                step_5["status"] = "success"
            except Exception as e:
                step_5["status"] = "error"
                step_5["error"] = str(e)
        else:
            step_5["status"] = "blocked"
            step_5["blocked_reason"] = f"User role '{user_role}' is not authorized to use tool 'generate_learning_plan'"
        
        trace.append(step_5)
        
        return {
            "employee_id": employee_id,
            "user_role": user_role,
            "trace": trace,
            "summary": {
                "total_steps": 5,
                "successful_steps": len([s for s in trace if s["status"] == "success"]),
                "blocked_steps": len([s for s in trace if s["status"] == "blocked"]),
                "error_steps": len([s for s in trace if s["status"] == "error"])
            }
        }
        
    except Exception as e:
        logger.exception("Workforce check failed")
        raise HTTPException(status_code=500, detail=str(e))

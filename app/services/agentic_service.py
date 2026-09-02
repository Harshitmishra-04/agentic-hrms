import pandas as pd
from typing import Dict, Any, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing_extensions import TypedDict

# Import existing services
from app.services import skill_gap_service, recommendation_service, attrition_service

# Define the state for the agent
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    employee_id: int
    context: Dict[str, Any]

# Tool authorization layer - enforced OUTSIDE the LLM
class ToolAuthorizer:
    """Authorizes tool execution based on governance rules."""
    
    # Permission matrix: which roles can use which tools
    PERMISSIONS = {
        "get_employee_profile": ["hr_admin", "manager", "employee"],
        "get_skills": ["hr_admin", "manager", "employee"],
        "get_role_requirements": ["hr_admin", "manager"],
        "calculate_skill_gap": ["hr_admin", "manager"],
        "recommend_courses": ["hr_admin", "manager", "employee"],
        "generate_learning_plan": ["hr_admin", "manager"],
    }
    
    @staticmethod
    def authorize(tool_name: str, user_role: str = "hr_admin") -> bool:
        """Check if a user role is authorized to use a tool."""
        if tool_name not in ToolAuthorizer.PERMISSIONS:
            return False
        return user_role in ToolAuthorizer.PERMISSIONS[tool_name]

# Tool implementations with inline authorization
@tool
def get_employee_profile(employee_id: int, user_role: str = "hr_admin") -> Dict[str, Any]:
    """Get the complete profile for an employee including demographics, risk, and engagement."""
    if not ToolAuthorizer.authorize("get_employee_profile", user_role):
        raise PermissionError(f"User role '{user_role}' is not authorized to use tool 'get_employee_profile'")
    
    profile = attrition_service.get_employee_intelligence(employee_id)
    if profile is None:
        return {"error": f"Employee {employee_id} not found"}
    return profile

@tool
def get_skills(employee_id: int, user_role: str = "hr_admin") -> Dict[str, Any]:
    """Get the current skills possessed by an employee."""
    if not ToolAuthorizer.authorize("get_skills", user_role):
        raise PermissionError(f"User role '{user_role}' is not authorized to use tool 'get_skills'")
    
    employee_skills = pd.read_csv("data/processed/employee_skills.csv")
    emp_skills = employee_skills[employee_skills["employee_id"] == employee_id]
    
    if emp_skills.empty:
        return {"error": f"No skills found for employee {employee_id}"}
    
    return {
        "employee_id": employee_id,
        "skills": emp_skills["current_skill"].tolist(),
        "skill_count": len(emp_skills)
    }

@tool
def get_role_requirements(employee_id: int, user_role: str = "hr_admin") -> Dict[str, Any]:
    """Get the role requirements for an employee's current role."""
    if not ToolAuthorizer.authorize("get_role_requirements", user_role):
        raise PermissionError(f"User role '{user_role}' is not authorized to use tool 'get_role_requirements'")
    
    # Get employee's role
    employees = pd.read_csv("data/processed/employees.csv")
    emp = employees[employees["EmployeeNumber"] == employee_id]
    
    if emp.empty:
        return {"error": f"Employee {employee_id} not found"}
    
    job_role = emp.iloc[0]["JobRole"]
    
    # Get role skills from the role_skills file (which uses O*NET titles, not JobRole)
    # For now, return a simplified response since the mapping is complex
    role_skills = pd.read_csv("data/processed/role_skills.csv")
    
    return {
        "employee_id": employee_id,
        "job_role": job_role,
        "note": "Role skills are mapped via O*NET titles, not directly via JobRole",
        "total_role_skills_available": len(role_skills),
        "skill_types": role_skills["Skill Type"].unique().tolist()
    }

@tool
def calculate_skill_gap(employee_id: int, user_role: str = "hr_admin") -> Dict[str, Any]:
    """Calculate the skill gap between an employee's current skills and role requirements."""
    if not ToolAuthorizer.authorize("calculate_skill_gap", user_role):
        raise PermissionError(f"User role '{user_role}' is not authorized to use tool 'calculate_skill_gap'")
    
    skill_gaps = pd.read_csv("data/processed/employee_skill_gaps.csv")
    emp_gaps = skill_gaps[skill_gaps["employee_id"] == employee_id]
    
    if emp_gaps.empty:
        return {"error": f"No skill gaps found for employee {employee_id}"}
    
    return {
        "employee_id": employee_id,
        "missing_skills": emp_gaps["missing_skill"].tolist(),
        "gap_count": len(emp_gaps),
        "avg_importance": emp_gaps["importance_score"].mean()
    }

@tool
def recommend_courses(employee_id: int, user_role: str = "hr_admin") -> Dict[str, Any]:
    """Get course recommendations for an employee based on their skill gaps."""
    if not ToolAuthorizer.authorize("recommend_courses", user_role):
        raise PermissionError(f"User role '{user_role}' is not authorized to use tool 'recommend_courses'")
    
    recommendations = pd.read_csv("data/processed/employee_course_recommendations_v3.csv")
    emp_recs = recommendations[recommendations["employee_id"] == employee_id]
    
    if emp_recs.empty:
        return {"error": f"No recommendations found for employee {employee_id}"}
    
    # Sort by rank
    emp_recs = emp_recs.sort_values("rank")
    
    return {
        "employee_id": employee_id,
        "recommendations": [
            {
                "rank": row["rank"],
                "course": row["recommended_course"],
                "target_skill": row["target_skill"],
                "cosine_score": row["cosine_score"]
            }
            for _, row in emp_recs.iterrows()
        ]
    }

@tool
def generate_learning_plan(employee_id: int, user_role: str = "hr_admin") -> Dict[str, Any]:
    """Generate a structured learning plan based on skill gaps and recommendations."""
    if not ToolAuthorizer.authorize("generate_learning_plan", user_role):
        raise PermissionError(f"User role '{user_role}' is not authorized to use tool 'generate_learning_plan'")
    
    # Get skill gaps directly (not using the tool wrapper)
    skill_gaps = pd.read_csv("data/processed/employee_skill_gaps.csv")
    emp_gaps = skill_gaps[skill_gaps["employee_id"] == employee_id]
    
    if emp_gaps.empty:
        return {"error": f"No skill gaps found for employee {employee_id}"}
    
    gaps_data = {
        "employee_id": employee_id,
        "missing_skills": emp_gaps["missing_skill"].tolist(),
        "gap_count": len(emp_gaps),
        "avg_importance": emp_gaps["importance_score"].mean()
    }
    
    # Get recommendations directly (not using the tool wrapper)
    recommendations = pd.read_csv("data/processed/employee_course_recommendations_v3.csv")
    emp_recs = recommendations[recommendations["employee_id"] == employee_id]
    
    if emp_recs.empty:
        return {"error": f"No recommendations found for employee {employee_id}"}
    
    # Sort by rank
    emp_recs = emp_recs.sort_values("rank")
    
    recs_data = {
        "employee_id": employee_id,
        "recommendations": [
            {
                "rank": row["rank"],
                "course": row["recommended_course"],
                "target_skill": row["target_skill"],
                "cosine_score": row["cosine_score"]
            }
            for _, row in emp_recs.iterrows()
        ]
    }
    
    # Create a prioritized learning plan
    learning_plan = {
        "employee_id": employee_id,
        "priority_gaps": gaps_data["missing_skills"][:5],  # Top 5 gaps
        "recommended_courses": recs_data["recommendations"],
        "estimated_completion_weeks": len(recs_data["recommendations"]) * 2,
        "plan_summary": f"Focus on closing {gaps_data['gap_count']} skill gaps through {len(recs_data['recommendations'])} recommended courses"
    }
    
    return learning_plan

# Create the tool list
tools = [
    get_employee_profile,
    get_skills,
    get_role_requirements,
    calculate_skill_gap,
    recommend_courses,
    generate_learning_plan,
]

# Create tool node
tool_node = ToolNode(tools)

# Simple agent nodes (placeholder for actual LLM integration)
def agent_node(state: AgentState):
    """Agent node that decides which tool to call (simplified - in production, this would use an LLM)."""
    messages = state["messages"]
    employee_id = state["employee_id"]
    user_role = state["context"].get("user_role", "hr_admin")
    
    # Simple routing logic based on the last message
    # In production, this would be an LLM decision
    last_message = messages[-1] if messages else {}
    query = last_message.get("content", "").lower() if isinstance(last_message, dict) else str(last_message).lower()
    
    # Determine which tool to call based on query
    if "profile" in query or "information" in query:
        tool_name = "get_employee_profile"
    elif "skills" in query and "current" in query:
        tool_name = "get_skills"
    elif "role" in query or "requirements" in query:
        tool_name = "get_role_requirements"
    elif "gap" in query or "missing" in query:
        tool_name = "calculate_skill_gap"
    elif "recommend" in query or "course" in query:
        tool_name = "recommend_courses"
    elif "plan" in query or "learning" in query:
        tool_name = "generate_learning_plan"
    else:
        # Default to learning plan
        tool_name = "generate_learning_plan"
    
    # Return tool call with user_role
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": {"employee_id": employee_id, "user_role": user_role},
                    "id": "tool_call_1"
                }]
            )
        ]
    }

def should_continue(state: AgentState):
    """Determine whether to continue or end the conversation."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return END

# Build the graph
def build_agentic_graph():
    """Build the LangGraph orchestrator with governed tool-calling."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()

# Global graph instance
_agentic_graph = None

def get_agentic_graph():
    """Get or create the agentic graph singleton."""
    global _agentic_graph
    if _agentic_graph is None:
        _agentic_graph = build_agentic_graph()
    return _agentic_graph

def run_agentic_workflow(employee_id: int, query: str, user_role: str = "hr_admin") -> Dict[str, Any]:
    """
    Run the agentic workflow for an employee query.
    
    Args:
        employee_id: The employee ID to analyze
        query: The user's query
        user_role: The role of the user making the request (for authorization)
    
    Returns:
        The final state with results
    """
    graph = get_agentic_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "employee_id": employee_id,
        "context": {"user_role": user_role}
    }
    
    result = graph.invoke(initial_state)
    
    return result

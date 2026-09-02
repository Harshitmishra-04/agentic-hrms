from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AttritionPredictRequest(BaseModel):
    """Request body for POST /predict/attrition.

    Field names match the employee master / attrition source columns.
    Extra keys are allowed so optional identity columns can pass through to logging.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True, str_strip_whitespace=True)

    Age: int = Field(..., ge=18, le=100)
    BusinessTravel: Literal["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
    DailyRate: int = Field(..., ge=0)
    Department: Literal["Human Resources", "Research & Development", "Sales"]
    DistanceFromHome: int = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EducationField: Literal[
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    ]
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: Literal["Female", "Male"]
    HourlyRate: int = Field(..., ge=0)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: Literal[
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ]
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: Literal["Divorced", "Married", "Single"]
    MonthlyIncome: int = Field(..., gt=0)
    MonthlyRate: int = Field(..., ge=0)
    NumCompaniesWorked: int = Field(..., ge=0)
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int = Field(..., ge=0)
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0)
    TrainingTimesLastYear: int = Field(..., ge=0, le=6)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0)
    YearsInCurrentRole: int = Field(..., ge=0)
    YearsSinceLastPromotion: int = Field(..., ge=0)
    YearsWithCurrManager: int = Field(..., ge=0)

    EmployeeNumber: Optional[int] = Field(default=None, gt=0)
    EmployeeCount: Optional[int] = None
    Over18: Optional[Literal["Y"]] = None
    StandardHours: Optional[int] = None
    Attrition: Optional[Literal["Yes", "No"]] = None
    engagement_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        alias="Engagement Score",
        description="Optional 1-5 engagement score; rejected when outside range.",
    )


class EngagementScoreRequest(BaseModel):
    """Standalone engagement-score payload used to reject invalid scores before service logic."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    engagement_score: int = Field(..., ge=1, le=5, alias="Engagement Score")
    satisfaction_score: Optional[int] = Field(default=None, ge=1, le=5, alias="Satisfaction Score")
    work_life_balance_score: Optional[int] = Field(
        default=None, ge=1, le=5, alias="Work-Life Balance Score"
    )


class EmployeeLookupRequest(BaseModel):
    """Path parameters for GET /employees/{employee_id}."""

    model_config = ConfigDict(extra="forbid")

    employee_id: int = Field(..., gt=0)


class DashboardQuery(BaseModel):
    """Query model for dashboard endpoints that take no filters."""

    model_config = ConfigDict(extra="forbid")


class SkillGapsQuery(BaseModel):
    """Query parameters for GET /dashboard/skill-gaps."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=15, ge=1, le=100)


class RAGQueryRequest(BaseModel):
    """Request body for POST /rag/ask."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500, description="User question about HR policies")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of relevant documents to retrieve")


class AgentWorkflowRequest(BaseModel):
    """Request body for POST /agent/workforce-check."""

    model_config = ConfigDict(extra="forbid")

    employee_id: int = Field(..., gt=0, description="Employee ID to analyze")
    user_role: Literal["hr_admin", "manager", "employee"] = Field(..., description="User role for authorization check")

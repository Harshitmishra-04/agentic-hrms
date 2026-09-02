import os
import subprocess
import time
import httpx
import signal

import pandas as pd
import streamlit as st
import plotly.express as px

# Start FastAPI backend in background process if not already running
_backend_process = None
_backend_started = False

def start_backend():
    """Start the FastAPI backend using uvicorn in a background subprocess."""
    import sys
    
    # Add the project root to the Python path so the backend can be imported
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Start uvicorn as a subprocess
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", 
               "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"]
        
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process
    except Exception as e:
        print(f"Failed to start backend subprocess: {e}")
        return None

def ensure_backend_running():
    """Ensure the FastAPI backend is running, starting it if necessary."""
    global _backend_process, _backend_started
    
    if _backend_started:
        return
    
    # Check if backend is already responding
    try:
        response = httpx.get("http://127.0.0.1:8000/", timeout=2.0)
        if response.status_code == 200:
            _backend_started = True
            return
    except:
        pass
    
    # Start backend in background subprocess
    if _backend_process is None or _backend_process.poll() is not None:
        _backend_process = start_backend()
        if _backend_process:
            _backend_started = True
            print(f"Started backend process with PID: {_backend_process.pid}")
        else:
            print("Failed to start backend process")

def wait_for_api_ready(max_retries=60, retry_interval=2.0):
    """Wait for the API to be ready before proceeding."""
    for attempt in range(max_retries):
        try:
            response = httpx.get("http://127.0.0.1:8000/", timeout=5.0)
            if response.status_code == 200:
                return True
        except Exception as e:
            # Log the error for debugging but continue trying
            if attempt % 10 == 0:  # Log every 10 attempts to avoid spam
                print(f"Attempt {attempt + 1}/{max_retries}: Backend not ready yet - {e}")
        time.sleep(retry_interval)
    return False

# Start backend on first script run
if "backend_started" not in st.session_state:
    ensure_backend_running()
    st.session_state.backend_started = True

# Cleanup function to kill backend process on exit
def cleanup_backend():
    global _backend_process
    if _backend_process and _backend_process.poll() is None:
        try:
            _backend_process.terminate()
            _backend_process.wait(timeout=5)
        except:
            try:
                _backend_process.kill()
            except:
                pass

# Register cleanup
import atexit
atexit.register(cleanup_backend)

# Wait for API to be ready before proceeding
if "api_ready" not in st.session_state:
    with st.spinner("Starting FastAPI backend... This may take up to 2 minutes."):
        if wait_for_api_ready(max_retries=60, retry_interval=2.0):
            st.session_state.api_ready = True
        else:
            st.error("Failed to start FastAPI backend. Please check the logs.")
            # Cleanup the failed process
            cleanup_backend()
            st.stop()

DEFAULT_API_URL = os.getenv("HRMS_API_URL", "http://127.0.0.1:8000")


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_json(base_url: str, path: str):
    with _client(base_url) as client:
        response = client.get(path)
        response.raise_for_status()
        return response.json()


@st.cache_data(show_spinner=False)
def fetch_employee(base_url: str, employee_id: int):
    with _client(base_url) as client:
        response = client.get(f"/employees/{employee_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def post_json(base_url: str, path: str, payload: dict):
    with _client(base_url) as client:
        response = client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


@st.cache_data(show_spinner=False)
def onet_basic_skill_names() -> set[str]:
    """O*NET Basic Skills are Element ID 2.A.* in essential_skills.csv.

    Element Name is the skill title (e.g. Critical Thinking), not a category.
    The category lives in Element ID: every row in essential_skills.csv is 2.A
    (Basic Skills). Software / specialized tools are not in that file.
    """
    path = os.path.join("data", "raw", "essential_skills.csv")
    if not os.path.exists(path):
        return set()
    ess = pd.read_csv(path, usecols=["Element ID", "Element Name"])
    basic = ess.loc[
        ess["Element ID"].astype(str).str.startswith("2.A"),
        "Element Name",
    ]
    return set(basic.dropna().unique())


def split_gaps_by_onet_family(gaps: list[dict], basic_names: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gaps_df = pd.DataFrame(gaps)
    if gaps_df.empty:
        empty = pd.DataFrame(columns=["skill", "missing_count", "avg_importance_score", "severity"])
        return empty, empty
    is_basic = gaps_df["skill"].isin(basic_names)
    return gaps_df.loc[is_basic].copy(), gaps_df.loc[~is_basic].copy()


def build_org_gap_analytics() -> pd.DataFrame:
    """Return org-wide skill demand vs availability by skill.

    Required = number of employees whose mapped role includes the skill.
    Available = number of employees who already have the skill in employee_skills.csv.
    Gap = Required - Available, with a simple hiring/reskilling split heuristic:
    60% of the gap should be reskilled internally and 40% hired externally.
    """
    emp_skills_path = os.path.join("data", "processed", "employee_skills.csv")
    emp_gaps_path = os.path.join("data", "processed", "employee_skill_gaps.csv")
    role_skills_path = os.path.join("data", "processed", "role_skills.csv")

    if not all(os.path.exists(p) for p in [emp_skills_path, emp_gaps_path, role_skills_path]):
        return pd.DataFrame(columns=["skill", "Required", "Available", "Gap", "Reskill (60%)", "Hire externally (40%)"])

    emp_skills = pd.read_csv(emp_skills_path)
    employee_roles = pd.read_csv(emp_gaps_path)[["employee_id", "onet_soc_code"]].drop_duplicates()
    role_skills = pd.read_csv(role_skills_path)

    required_counts = {}
    for _, row in employee_roles.iterrows():
        onet = row["onet_soc_code"]
        req_skills = role_skills.loc[role_skills["O*NET-SOC Code"] == onet, "Skill Name"].tolist()
        for skill in req_skills:
            required_counts[skill] = required_counts.get(skill, 0) + 1

    available_counts = emp_skills["current_skill"].value_counts().to_dict()
    rows = []
    for skill, required in required_counts.items():
        available = int(available_counts.get(skill, 0))
        gap = max(required - available, 0)
        rows.append({
            "skill": skill,
            "Required": int(required),
            "Available": int(available),
            "Gap": int(gap),
            "Reskill (60%)": int(round(gap * 0.6)),
            "Hire externally (40%)": int(round(gap * 0.4)),
        })

    if not rows:
        return pd.DataFrame(columns=["skill", "Required", "Available", "Gap", "Reskill (60%)", "Hire externally (40%)"])

    return pd.DataFrame(rows).sort_values("Gap", ascending=False).reset_index(drop=True)


def get_employee_current_role(employee_id: int) -> tuple[str | None, str | None, str | None]:
    gap_path = os.path.join("data", "processed", "employee_skill_gaps.csv")
    if not os.path.exists(gap_path):
        return None, None, None
    emp_gaps = pd.read_csv(gap_path)
    emp_row = emp_gaps[emp_gaps["employee_id"] == employee_id]
    if emp_row.empty:
        return None, None, None
    current_role = emp_row.iloc[0].get("job_role")
    onet_soc_code = emp_row.iloc[0].get("onet_soc_code")
    return current_role, onet_soc_code, str(onet_soc_code)


def compute_target_role_readiness(employee_id: int, target_onet: str) -> dict:
    emp_skills_path = os.path.join("data", "processed", "employee_skills.csv")
    role_skills_path = os.path.join("data", "processed", "role_skills.csv")
    role_reference_path = os.path.join("data", "processed", "role_reference.csv")

    if not all(os.path.exists(p) for p in [emp_skills_path, role_skills_path, role_reference_path]):
        return {"error": "Required files missing for career-path simulation."}

    emp_skills = pd.read_csv(emp_skills_path)
    role_skills = pd.read_csv(role_skills_path)
    role_reference = pd.read_csv(role_reference_path)
    target_role = role_reference[role_reference["onet_soc_code"] == target_onet]
    if target_role.empty:
        return {"error": f"Target role {target_onet} was not found in role_reference.csv."}

    current_role_name, current_onet, _ = get_employee_current_role(employee_id)
    current_skills = set(
        emp_skills.loc[emp_skills["employee_id"] == employee_id, "current_skill"].dropna().astype(str).tolist()
    )
    required = role_skills.loc[role_skills["O*NET-SOC Code"] == target_onet].copy()
    if required.empty:
        return {"error": f"No skill requirements are available for target O*NET code {target_onet}."}

    total_required = float(required["Relevance Score"].sum())
    matched = float(required[required["Skill Name"].isin(current_skills)]["Relevance Score"].sum())
    readiness_today = (matched / total_required * 100) if total_required > 0 else 0.0

    missing = required.loc[~required["Skill Name"].isin(current_skills)].copy()
    missing = missing.sort_values("Relevance Score", ascending=False).head(3)
    projected_gain = float(missing["Relevance Score"].sum())
    projected_after = min(((matched + projected_gain) / total_required * 100) if total_required > 0 else 0.0, 100.0)

    return {
        "employee_id": employee_id,
        "current_role": current_role_name,
        "current_onet": current_onet,
        "target_role": target_role.iloc[0]["role_title"],
        "target_onet": target_onet,
        "readiness_today": float(readiness_today),
        "projected_after": float(projected_after),
        "current_skill_count": len(current_skills),
        "required_skill_count": int(len(required)),
        "missing_skill_count": int(len(missing)),
        "missing_skills": missing["Skill Name"].tolist(),
        "missing_skill_importance": [float(x) for x in missing["Relevance Score"].tolist()],
    }


st.set_page_config(
    page_title="Agentic HRMS — Workforce Intelligence",
    page_icon="📊",
    layout="wide",
)

# Keep the dashboard compact: avoid global card wrappers and extra visual gutters.
st.markdown("""
<style>
/* Remove the big extra borders/gaps caused by generic Streamlit blocks. */
div[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

/* KPI card coloring - Light theme with pastel backgrounds and dark text */
.kpi-card {
    background-color: #f0f9ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
}

/* Light theme text colors */
.kpi-label {
    color: #475569 !important;
}

.kpi-value {
    color: #0c4a6e !important;
}

.caption-muted {
    color: #64748b !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Workforce Intelligence Dashboard")
st.caption("Live view of attrition risk, skill gaps, and course recommendations via the FastAPI backend.")
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Ask HR Policy", "Workforce Intelligence Agent", "Career Path Simulation"])

with tab1:
    with st.sidebar:
        st.header("⚙️ Controls")
        api_url = st.text_input("API base URL", value=DEFAULT_API_URL)
        st.caption("Start the backend with `uvicorn app.main:app --reload` before loading this page.")

    try:
        summary = fetch_json(api_url, "/dashboard/summary")
        dept_rows = fetch_json(api_url, "/dashboard/attrition-by-department")
        skill_payload = fetch_json(api_url, "/dashboard/skill-gaps?limit=100")
        recs_payload = fetch_json(api_url, "/dashboard/recommendations")
    except httpx.HTTPError as exc:
        st.error(
            f"Could not reach the FastAPI backend at `{api_url}`. "
            "Start it from the project root, then refresh this page."
        )
        st.code(str(exc))
        st.stop()

    dept_df = pd.DataFrame(dept_rows)
    departments = ["All departments"]
    if not dept_df.empty:
        departments += sorted(dept_df["department"].tolist())

    with st.sidebar:
        selected_dept = st.selectbox("Department filter", departments)

    filtered_dept_df = dept_df.copy()
    if selected_dept != "All departments" and not dept_df.empty:
        filtered_dept_df = dept_df[dept_df["department"] == selected_dept]

    attrition = summary.get("attrition", {})
    engagement = summary.get("engagement", {})
    skills = summary.get("skills", {})
    recommendations = summary.get("recommendations", {})
    risk_counts = attrition.get("risk_counts", {}) or {}

    if selected_dept != "All departments" and not filtered_dept_df.empty:
        row = filtered_dept_df.iloc[0]
        kpi_employees = int(row["total_employees"])
        kpi_avg_prob = float(row["average_probability"])
        dept_risk = row.get("risk_distribution") or {}
        kpi_high = int(dept_risk.get("High", 0))
    else:
        kpi_employees = int(attrition.get("total_employees", 0))
        kpi_avg_prob = float(attrition.get("average_probability", 0.0))
        kpi_high = int(risk_counts.get("High", 0))

    kpi_engagement = engagement.get("avg_engagement_score")
    kpi_high_gaps = (skills.get("severity_distribution") or {}).get("HIGH", 0)
    kpi_recs = int(recommendations.get("total_recommendations", 0))

    # KPI block
    with st.container():
        st.subheader("KPI snapshot — employee master")
        st.caption(
            "These figures come from the attrition / employee-master population "
            f"(`employee_intelligence.csv`, n={attrition.get('total_employees', kpi_employees)}). "
            + (
                f"Attrition KPIs are scoped to **{selected_dept}**."
                if selected_dept != "All departments"
                else "They are **not** joined to the engagement survey."
            )
        )
        st.caption("High-risk = attrition probability >= 0.40 (the model's chosen operating threshold). HIGH skill gap severity = 100+ employees missing that skill (MEDIUM 50-99, LOW <50).")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="kpi-card" style="background-color: #dbeafe; border: 1px solid #93c5fd; border-radius: 8px; padding: 15px; margin: 5px 0; text-align: center;"><div style="font-size: 24px;">👥</div><div style="font-size: 12px; color: #475569; margin-top: 5px;">Employees</div><div style="font-size: 24px; font-weight: bold; color: #0c4a6e; margin-top: 10px;">{kpi_employees:,}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card" style="background-color: #d1fae5; border: 1px solid #6ee7b7; border-radius: 8px; padding: 15px; margin: 5px 0; text-align: center;"><div style="font-size: 24px;">⚠️</div><div style="font-size: 12px; color: #0d504c; margin-top: 5px;">Avg attrition risk</div><div style="font-size: 24px; font-weight: bold; color: #065f46; margin-top: 10px;">{kpi_avg_prob:.1%}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card" style="background-color: #fed7aa; border: 1px solid #fdba74; border-radius: 8px; padding: 15px; margin: 5px 0; text-align: center;"><div style="font-size: 24px;">🔴</div><div style="font-size: 12px; color: #5e4d3b; margin-top: 5px;">High-risk employees</div><div style="font-size: 24px; font-weight: bold; color: #92400e; margin-top: 10px;">{kpi_high:,}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card" style="background-color: #e0e7ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: 15px; margin: 5px 0; text-align: center;"><div style="font-size: 24px;">📊</div><div style="font-size: 12px; color: #3730a3; margin-top: 5px;">HIGH skill gaps</div><div style="font-size: 24px; font-weight: bold; color: #3730a3; margin-top: 10px;">{int(kpi_high_gaps):,}</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="kpi-card" style="background-color: #fce7f3; border: 1px solid #fbcfe8; border-radius: 8px; padding: 15px; margin: 5px 0; text-align: center;"><div style="font-size: 24px;">🎓</div><div style="font-size: 12px; color: #500724; margin-top: 5px;">Course recommendations</div><div style="font-size: 24px; font-weight: bold; color: #831843; margin-top: 10px;">{kpi_recs:,}</div></div>', unsafe_allow_html=True)

    # Engagement block
    with st.container():
        eng_n = int(engagement.get("total_records", 0) or 0)
        st.subheader("Engagement survey — separate population")
        st.caption(
            f"Avg engagement (survey pop., n={eng_n:,}). "
            "This is a separate dataset, **not** joined to the 1,470 employee master."
        )
        with st.expander("ℹ️ How this is calculated"):
            st.caption(
                "Source: `GET /dashboard/summary` → `engagement.avg_engagement_score` = "
                "mean(`Engagement Score`) on `data/processed/engagement_data.csv` "
                f"(n={eng_n:,} survey records). This is **not** computed from "
                "`employee_intelligence.csv` and is **not** the average of the 1,470 employee-master rows."
            )
        st.metric(
            f"Avg engagement (survey pop., n={eng_n:,})",
            f"{kpi_engagement:.2f}" if kpi_engagement is not None else "n/a",
        )

    # Charts block
    with st.container():
        chart_col, gap_col = st.columns(2)

        with chart_col:
            st.subheader("Attrition risk by department")
            st.caption("Bar height/count = headcount at each risk level, not risk severity - see avg_risk_pct for the actual per-department comparison.")
            if filtered_dept_df.empty:
                st.info("No department attrition data returned.")
            else:
                chart_source = filtered_dept_df.copy()
                
                # Overall risk distribution donut chart (from overall attrition data)
                overall_risk_df = pd.DataFrame([
                    {"Risk Level": "High", "Count": risk_counts.get("High", 0)},
                    {"Risk Level": "Medium", "Count": risk_counts.get("Medium", 0)},
                    {"Risk Level": "Low", "Count": risk_counts.get("Low", 0)}
                ])
                overall_risk_df = overall_risk_df[overall_risk_df["Count"] > 0]

                fig_pie = px.pie(
                    overall_risk_df,
                    values="Count",
                    names="Risk Level",
                    hole=0.5,
                    color="Risk Level",
                    color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                    title="Overall Risk Distribution"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    template="plotly_white",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=30, b=30, l=30, r=30),
                    font=dict(color="#1a1a1a"),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8f9fa"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Keep a single compact per-department avg_risk_pct view.
                # The stacked headcount chart is intentionally removed to avoid
                # showing the same risk data in three parallel forms.
                dept_risk_view = chart_source[["department", "average_probability"]].copy()
                dept_risk_view["avg_risk_pct"] = (dept_risk_view["average_probability"] * 100).round(1)
                dept_risk_view = dept_risk_view[["department", "avg_risk_pct"]]
                st.dataframe(dept_risk_view, hide_index=True, width="stretch")

        with gap_col:
            st.subheader("Org-wide skill gap view")
            st.caption("Required vs Available skills across the org. Gap = Required - Available.")
            with st.expander("ℹ️ How gaps are split"):
                st.caption(
                    "For workforce planning: Required = employees whose mapped role requires the skill; "
                    "Available = employees who already have it; Gap = Required - Available. "
                    "The 60% / 40% split suggests 60% of the gap should be addressed via internal reskilling "
                    "and 40% via external hiring."
                )
            org_gap_df = build_org_gap_analytics()
            if org_gap_df.empty:
                st.info("No org-wide gap data returned.")
            else:
                org_gap_df = org_gap_df[["skill", "Required", "Available", "Gap", "Reskill (60%)", "Hire externally (40%)"]]
                gap_chart_df = org_gap_df.head(15).copy()
                st.bar_chart(gap_chart_df.set_index("skill")["Gap"], y_label="Gap (Required - Available)")
                st.dataframe(org_gap_df.head(15), hide_index=True, width="stretch")
                with st.expander("View full org-wide gap table"):
                    st.dataframe(org_gap_df, hide_index=True, width="stretch")

            # Preserve the legacy top-skill summary as a secondary detail if available.
            top_skills = skill_payload.get("top_missing_skills") or []
            if top_skills:
                st.caption("Legacy top missing-skill summary (for reference only):")
                legacy_df = pd.DataFrame(top_skills)[["skill", "missing_count", "avg_importance_score", "severity"]]
                st.dataframe(legacy_df.head(10), hide_index=True, width="stretch")

    # Course catalog block
    with st.container():
        st.subheader("Course recommendation catalog")
        st.caption("times_recommended = number of employees this course was suggested to (not completions).")
        catalog = recs_payload.get("catalog") or []
        distribution = (recs_payload.get("summary") or {}).get("course_distribution") or {}
        if catalog:
            recs_df = pd.DataFrame(catalog)
            recs_df["times_recommended"] = recs_df["course_title"].map(distribution).fillna(0).astype(int)
            recs_df = recs_df.sort_values("times_recommended", ascending=False)
            st.dataframe(recs_df.head(10), hide_index=True, width="stretch")
            with st.expander("View all courses"):
                st.dataframe(recs_df, hide_index=True, width="stretch")
        else:
            st.info("No course catalog returned.")

    # Employee drill-down block
    with st.container():
        st.subheader("Employee drill-down")
        drill_col, detail_col = st.columns([1, 3])
        with drill_col:
            employee_id = st.number_input("Employee ID", min_value=1, step=1, value=1, key="drilldown_employee_id")
            st.caption("IDs are EmployeeNumber values and are **not sequential** (e.g. 3, 6, 9 are unused).")
            load_profile = st.button("Load profile", type="primary")

        if load_profile:
            try:
                record = fetch_employee(api_url, int(employee_id))
            except httpx.HTTPError as exc:
                st.error(f"Employee lookup failed: {exc}")
                record = None

            if record is None:
                st.warning(
                    f"Employee ID {int(employee_id)} was not found in the employee master "
                    "(HTTP 404). This is expected when the ID has a gap in `EmployeeNumber` "
                    "— it is not a lookup crash."
                )
            else:
                if selected_dept != "All departments" and record.get("Department") != selected_dept:
                    st.info(
                        f"This employee is in **{record.get('Department')}**, "
                        f"which is outside the current department filter (**{selected_dept}**)."
                    )
                with detail_col:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Role", record.get("JobRole") or "n/a")
                    m2.metric("Department", record.get("Department") or "n/a")
                    probability = record.get("attrition_probability")
                    m3.metric(
                        "Attrition probability",
                        f"{probability:.1%}" if probability is not None else "n/a",
                    )
                    
                    # Risk bucket as colored badge
                    risk_bucket = record.get("risk_bucket")
                    if risk_bucket == "High":
                        m4.markdown("🔴 **High**")
                    elif risk_bucket == "Medium":
                        m4.markdown("🟡 **Medium**")
                    elif risk_bucket == "Low":
                        m4.markdown("🟢 **Low**")
                    else:
                        m4.metric("Risk bucket", risk_bucket or "n/a")

                eng = record.get("avg_engagement_score")
                skill_gap_count = record.get('total_skill_gap_count', 'n/a')
                
                # Calculate org average skill gap count
                org_avg_gaps = skills.get("avg_skill_gap_count", 0)
                
                # Calculate readiness scores
                readiness_path = os.path.join("data", "processed", "employee_skill_gaps.csv")
                role_skills_path = os.path.join("data", "processed", "role_skills.csv")
                emp_skills_path = os.path.join("data", "processed", "employee_skills.csv")
                recs_path = os.path.join("data", "processed", "employee_recommendations_pivoted_v3.csv")
                courses_path = os.path.join("data", "processed", "courses.csv")
                
                readiness_today = 0.0
                projected_readiness = 0.0
                
                if os.path.exists(readiness_path) and os.path.exists(role_skills_path) and os.path.exists(emp_skills_path):
                    try:
                        # Get employee's O*NET-SOC code and missing skills
                        emp_gaps_df = pd.read_csv(readiness_path)
                        emp_gaps = emp_gaps_df[emp_gaps_df["employee_id"] == int(employee_id)]
                        
                        if not emp_gaps.empty:
                            onet_soc_code = emp_gaps.iloc[0]["onet_soc_code"]
                            missing_skills_with_importance = dict(zip(emp_gaps["missing_skill"], emp_gaps["importance_score"]))
                            
                            # Get all required skills for this role
                            role_skills_df = pd.read_csv(role_skills_path)
                            role_required = role_skills_df[role_skills_df["O*NET-SOC Code"] == onet_soc_code]
                            
                            if not role_required.empty:
                                total_required_importance = role_required["Relevance Score"].sum()
                                
                                # Get employee's current skills
                                emp_skills_df = pd.read_csv(emp_skills_path)
                                emp_current_skills = emp_skills_df[emp_skills_df["employee_id"] == int(employee_id)]["current_skill"].tolist()
                                
                                # Calculate weighted readiness today
                                matched_importance = 0.0
                                for _, req_skill in role_required.iterrows():
                                    skill_name = req_skill["Skill Name"]
                                    importance = req_skill["Relevance Score"]
                                    if skill_name in emp_current_skills:
                                        matched_importance += importance
                                
                                readiness_today = (matched_importance / total_required_importance * 100) if total_required_importance > 0 else 0.0
                                
                                # Get target skills from recommended courses
                                if os.path.exists(recs_path) and os.path.exists(courses_path):
                                    recs_df = pd.read_csv(recs_path)
                                    courses_df = pd.read_csv(courses_path)
                                    
                                    emp_recs = recs_df[recs_df["employee_id"] == int(employee_id)]
                                    if not emp_recs.empty:
                                        rec_row = emp_recs.iloc[0]
                                        course_titles = [
                                            rec_row.get("recommended_course_1"),
                                            rec_row.get("recommended_course_2"),
                                            rec_row.get("recommended_course_3")
                                        ]
                                        course_titles = [c for c in course_titles if pd.notna(c) and c != ""]
                                        
                                        # Get target skills from courses
                                        target_skills = []
                                        for title in course_titles:
                                            course_match = courses_df[courses_df["course_title"] == title]
                                            if not course_match.empty:
                                                target_skills.append(course_match.iloc[0]["target_skill"])
                                        
                                        # Calculate projected readiness (assume target skills are acquired)
                                        projected_matched_importance = matched_importance
                                        for target_skill in target_skills:
                                            # Find importance of this target skill in role requirements
                                            skill_importance = role_required[role_required["Skill Name"] == target_skill]["Relevance Score"]
                                            if not skill_importance.empty:
                                                projected_matched_importance += skill_importance.iloc[0]
                                        
                                        projected_readiness = (projected_matched_importance / total_required_importance * 100) if total_required_importance > 0 else 0.0
                                        projected_readiness = min(projected_readiness, 100.0)  # Cap at 100%
                    except Exception as e:
                        st.warning(f"Could not calculate readiness scores: {e}")
                
                st.write(
                    f"**Engagement score:** {eng if eng is not None else 'not linked (separate engagement dataset)'}  \n"
                    f"**Skill gap count:** {skill_gap_count} (org average: {org_avg_gaps:.1f})"
                )
                
                # Readiness scoring section
                if readiness_today > 0 or projected_readiness > 0:
                    st.write("**Readiness scoring**")
                    r1, r2 = st.columns(2)
                    r1.metric("Readiness today", f"{readiness_today:.1f}%")
                    r2.metric("Projected after plan", f"{projected_readiness:.1f}%")
                    
                    # Before/after bar chart
                    readiness_df = pd.DataFrame({
                        "Scenario": ["Today", "After completing courses"],
                        "Readiness %": [readiness_today, projected_readiness]
                    })
                    fig_readiness = px.bar(
                        readiness_df,
                        x="Scenario",
                        y="Readiness %",
                        color="Scenario",
                        color_discrete_map={"Today": "#94a3b8", "After completing courses": "#10b981"},
                        title="Readiness improvement projection"
                    )
                    fig_readiness.update_layout(
                        template="plotly_white",
                        showlegend=False,
                        yaxis_range=[0, 100],
                        font=dict(color="#1a1a1a"),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#f8f9fa"
                    )
                    st.plotly_chart(fig_readiness, use_container_width=True)
                
                # Course recommendations - show all 3 (reuse paths from readiness calculation)
                if os.path.exists(recs_path) and os.path.exists(courses_path):
                    recs_df = pd.read_csv(recs_path)
                    courses_df = pd.read_csv(courses_path)
                    
                    emp_recs = recs_df[recs_df["employee_id"] == int(employee_id)]
                    if not emp_recs.empty:
                        rec_row = emp_recs.iloc[0]
                        course_titles = [
                            rec_row.get("recommended_course_1"),
                            rec_row.get("recommended_course_2"),
                            rec_row.get("recommended_course_3")
                        ]
                        course_titles = [c for c in course_titles if pd.notna(c) and c != ""]
                        
                        if course_titles:
                            # Look up course details
                            rec_details = []
                            for title in course_titles:
                                course_match = courses_df[courses_df["course_title"] == title]
                                if not course_match.empty:
                                    rec_details.append({
                                        "course_title": title,
                                        "target_skill": course_match.iloc[0]["target_skill"],
                                        "difficulty": course_match.iloc[0]["difficulty"]
                                    })
                                else:
                                    rec_details.append({
                                        "course_title": title,
                                        "target_skill": "n/a",
                                        "difficulty": "n/a"
                                    })
                            
                            st.write("**Course recommendations**")
                            st.dataframe(pd.DataFrame(rec_details), hide_index=True, width="stretch")
                
                # Missing skills section
                gaps = record.get("skill_gap_list")
                if gaps:
                    gap_items = [item.strip() for item in str(gaps).split(",") if item.strip()]
                    st.write("**Missing skills**")
                    st.dataframe(pd.DataFrame({"missing_skill": gap_items[:10]}), hide_index=True, width="stretch")
                    with st.expander("View all missing skills"):
                        st.dataframe(pd.DataFrame({"missing_skill": gap_items}), hide_index=True, width="stretch")

with tab2:
    st.subheader("Ask HR Policy (RAG)")
    st.caption(
        "Answers HR policy questions using hybrid retrieval (BM25 + vector search) with cross-encoder reranking. "
        "Answer generated by an LLM (via OpenRouter), grounded in the retrieved policy excerpts shown below."
    )
    
    rag_query = st.text_input("Your question about HR policies", placeholder="e.g., What is the company's leave policy?")
    top_k = st.slider("Number of relevant documents to retrieve", min_value=1, max_value=10, value=3)
    
    if st.button("Ask", type="primary"):
        if not rag_query:
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("🔍 Retrieving policies... This may take a few seconds."):
                    result = post_json(api_url, "/rag/ask", {"query": rag_query, "top_k": top_k})
                
                st.subheader("Answer")
                answer = result.get("answer", "No answer generated.")
                # Remove redundant template response disclaimer if present
                if "Note: This is a template response" in answer:
                    # Remove everything from "Note:" onwards
                    answer = answer.split("Note:")[0].strip()
                elif "In production, an LLM would generate" in answer:
                    # Alternative template response format
                    answer = answer.split("In production")[0].strip()
                st.write(answer)
                
                # Display timing breakdown
                timing = result.get("timing", {})
                if timing:
                    with st.expander("⏱️ Performance timing"):
                        timing_col1, timing_col2, timing_col3 = st.columns(3)
                        with timing_col1:
                            st.metric("Hybrid Search", f"{timing.get('hybrid_search', 0):.2f}s")
                        with timing_col2:
                            st.metric("Reranking", f"{timing.get('rerank', 0):.2f}s")
                        with timing_col3:
                            st.metric("LLM Answer", f"{timing.get('llm_answer', 0):.2f}s")
                        st.metric("Total", f"{timing.get('total', 0):.2f}s", delta=None)
                
                st.subheader("Retrieved Sources")
                sources = result.get("sources", [])
                if sources:
                    for i, source in enumerate(sources, 1):
                        score = source.get('score', 0)
                        file_name = source.get('metadata', {}).get('source_file', 'Unknown')
                        document = source.get('document', '')
                        
                        # Trim document to whole word boundaries
                        if document:
                            # Trim leading partial word by splitting on first space
                            if document and not document[0].isalnum():
                                parts = document.split(maxsplit=1)
                                document = parts[1] if len(parts) > 1 else document
                            
                            # Trim trailing partial word by splitting on last space
                            if document and not document[-1].isalnum():
                                parts = document.rsplit(maxsplit=1)
                                document = parts[0] if len(parts) > 1 else document
                        
                        with st.expander(f"Source {i} (score: {score:.3f})"):
                            st.caption(f"📄 {file_name}")
                            st.text_area("Document excerpt", document, height=150, key=f"rag_source_{i}")
                else:
                    st.info("No sources retrieved.")
                    
            except httpx.HTTPError as exc:
                st.error(f"RAG query failed: {exc}")

with tab4:
    st.subheader("Career Path Simulation")
    st.caption(
        "Pick an employee and a target role. The dashboard compares the employee's current-role skills to the target-role requirements, then shows readiness before and after completing the recommended course plan."
    )
    role_ref = pd.read_csv(os.path.join("data", "processed", "role_reference.csv"))
    role_options = role_ref[["onet_soc_code", "role_title"]].drop_duplicates().sort_values("role_title")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_employee_id = st.number_input("Employee ID", min_value=1, step=1, value=1, key="sim_employee_id")
    with sim_col2:
        sim_target_title = st.selectbox("Target role", role_options["role_title"].tolist(), index=role_options["role_title"].tolist().index("Sales Engineers") if "Sales Engineers" in role_options["role_title"].tolist() else 0)

    sim_target_onet = role_options.loc[role_options["role_title"] == sim_target_title, "onet_soc_code"].iloc[0]
    if st.button("Run career path simulation", type="primary"):
        result = compute_target_role_readiness(int(sim_employee_id), str(sim_target_onet))
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"Target role: {result['target_role']} ({result['target_onet']})")
            st.write(f"Current role: {result['current_role']} ({result['current_onet']})")
            readiness_df = pd.DataFrame({
                "Scenario": ["Current role readiness", "After recommended course plan"],
                "Readiness %": [result["readiness_today"], result["projected_after"]],
            })
            fig = px.bar(
                readiness_df,
                x="Scenario",
                y="Readiness %",
                color="Scenario",
                color_discrete_map={"Current role readiness": "#94a3b8", "After recommended course plan": "#10b981"},
                title="Readiness before vs after completing the recommended gap plan",
            )
            fig.update_layout(
                template="plotly_white",
                showlegend=False,
                yaxis_range=[0, 100],
                font=dict(color="#1a1a1a"),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8f9fa"
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Readiness today", f"{result['readiness_today']:.1f}%")
            c2.metric("Projected after plan", f"{result['projected_after']:.1f}%")

            st.write("**Skills that remain missing for the target role**")
            missing = pd.DataFrame({
                "missing_skill": result["missing_skills"],
                "importance": result["missing_skill_importance"],
            })
            st.dataframe(missing, hide_index=True, width="stretch")

            st.write("**Current-role vs target-role comparison**")
            current_skills = set(pd.read_csv(os.path.join("data", "processed", "employee_skills.csv")).loc[
                lambda d: d["employee_id"] == int(sim_employee_id), "current_skill"
            ].dropna().astype(str).tolist())
            target_required = pd.read_csv(os.path.join("data", "processed", "role_skills.csv"))
            target_required = target_required[target_required["O*NET-SOC Code"] == sim_target_onet]
            current_role = pd.read_csv(os.path.join("data", "processed", "employee_skill_gaps.csv"))
            current_role = current_role[current_role["employee_id"] == int(sim_employee_id)]
            current_role_name = current_role["job_role"].iloc[0] if not current_role.empty else "Unknown"
            st.write(f"Employee {sim_employee_id} is currently in {current_role_name} and is evaluating a move to {result['target_role']}.")
            st.write(f"Current skills held: {len(current_skills)}")
            st.write(f"Target-role requirements: {len(target_required)} skills")

with tab3:
    st.subheader("Workforce Intelligence Agent")
    st.caption(
        "Runs a 5-step tool sequence for workforce intelligence analysis. "
        "**Note:** Agent routing is currently simplified/rule-based, not full LLM-driven tool selection. "
        "Tool authorization is enforced based on user role."
    )
    
    agent_col1, agent_col2 = st.columns(2)
    with agent_col1:
        agent_employee_id = st.number_input("Employee ID", min_value=1, step=1, value=1, key="agent_employee_id")
    with agent_col2:
        agent_user_role = st.selectbox("Your role", ["hr_admin", "manager", "employee"])
    
    if st.button("Run Workforce Check", type="primary"):
        try:
            result = post_json(api_url, "/agent/workforce-check", {
                "employee_id": agent_employee_id,
                "user_role": agent_user_role
            })
            
            st.subheader("Workflow Summary")
            summary = result.get("summary", {})
            st.write(f"Total steps: {summary.get('total_steps', 0)}")
            st.write(f"Successful: {summary.get('successful_steps', 0)}")
            st.write(f"Blocked: {summary.get('blocked_steps', 0)}")
            st.write(f"Errors: {summary.get('error_steps', 0)}")
            
            st.subheader("Step-by-Step Trace")
            trace = result.get("trace", [])
            
            # Find first blocked step if any
            first_blocked_idx = None
            for i, step in enumerate(trace):
                if step.get("status") == "blocked":
                    first_blocked_idx = i
                    break
            
            for i, step in enumerate(trace):
                step_status = step.get("status", "unknown")
                step_num = step.get("step", 0)
                tool_name = step.get("tool", "unknown")
                
                status_color = {
                    "success": "🟢",
                    "blocked": "🔴",
                    "error": "⚠️",
                    "pending": "⏳"
                }.get(step_status, "❓")
                
                # Auto-expand first blocked step, otherwise collapse all
                auto_expand = (i == first_blocked_idx)
                
                with st.expander(f"Step {step_num}: {tool_name} {status_color} ({step_status.upper()})", expanded=auto_expand):
                    # Plain-language input summary
                    inputs = step.get("inputs", {})
                    emp_id = inputs.get("employee_id", "N/A")
                    user_role = inputs.get("user_role", "N/A")
                    st.write(f"**Input:** Checking employee #{emp_id} as role: {user_role}")
                    
                    # Raw inputs in collapsed expander
                    with st.expander("Show raw input", expanded=False):
                        st.json(inputs)
                    
                    if step_status == "blocked":
                        st.error(f"Blocked: {step.get('blocked_reason', 'No reason provided')}")
                    elif step_status == "error":
                        st.error(f"Error: {step.get('error', 'No error details')}")
                    elif step_status == "success":
                        output = step.get("output", {})
                        
                        # Human-readable summary based on tool type
                        if tool_name == "get_employee_profile":
                            job_role = output.get("JobRole", "N/A")
                            dept = output.get("Department", "N/A")
                            risk = output.get("risk_bucket", "N/A")
                            gap_count = output.get("total_skill_gap_count", "N/A")
                            st.write(f"**Result:** {job_role}, {dept} department, Risk: {risk}, {gap_count} skill gaps")
                        elif tool_name == "get_skills":
                            skill_count = output.get("skill_count", "N/A")
                            st.write(f"**Result:** Found {skill_count} current skills")
                        elif tool_name == "calculate_skill_gap":
                            gap_count = output.get("gap_count", "N/A")
                            avg_importance = output.get("avg_importance", "N/A")
                            st.write(f"**Result:** {gap_count} missing skills, avg importance: {avg_importance:.2f}")
                        elif tool_name == "recommend_courses":
                            rec_count = len(output.get("recommendations", []))
                            st.write(f"**Result:** {rec_count} course recommendations")
                        elif tool_name == "generate_learning_plan":
                            plan_summary = output.get("plan_summary", "N/A")
                            weeks = output.get("estimated_completion_weeks", "N/A")
                            st.write(f"**Result:** {plan_summary} (estimated {weeks} weeks)")
                        else:
                            st.write(f"**Result:** Tool executed successfully")
                        
                        # Full output in collapsed expander
                        with st.expander("Show raw output", expanded=False):
                            st.json(output)
                    else:
                        st.info(f"Status: {step_status}")
                        
        except httpx.HTTPError as exc:
            st.error(f"Agent workflow failed: {exc}")

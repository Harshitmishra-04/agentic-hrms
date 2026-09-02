# Enterprise HR AI — Agentic HRMS

An AI-powered Human Resource Management System that combines machine learning, RAG (Retrieval-Augmented Generation), and agentic workflows to deliver workforce intelligence, attrition prediction, skill gap analysis, and policy-aware HR assistance.

## Features

- **Attrition Prediction** — ML model (Logistic Regression + threshold tuning) predicts employee attrition risk with SHAP explainability
- **Skill Gap Engine** — Maps employee skills against O*NET role requirements to identify individual and organizational skill gaps
- **Recommendation Engine** — Semantic course recommendations using sentence-transformers embeddings and cosine similarity
- **RAG-Powered HR Policy QA** — Hybrid retrieval (BM25 + vector search) with cross-encoder reranking and LLM-grounded answers
- **Agentic Workforce Intelligence** — LangGraph-based agent with governed tool-calling for employee profiling, skill analysis, and learning plan generation
- **Recruitment Agent** — Resume-to-job description matching using skill embedding similarity
- **Interactive Dashboard** — Streamlit frontend with KPI cards, department filters, attrition charts, skill gap views, and employee drill-down

## Project Structure

```
enterprise_hr_ai/
├── app/                          # Application source code
│   ├── main.py                   # FastAPI application entry point
│   ├── api/
│   │   └── endpoints.py          # REST API endpoints
│   ├── ml/
│   │   ├── model_loader.py       # Model caching and loading
│   │   └── predictor.py          # Attrition prediction logic
│   ├── services/
│   │   ├── attrition_service.py  # Attrition risk business logic
│   │   ├── engagement_service.py # Engagement analytics
│   │   ├── skill_gap_service.py  # Skill gap calculations
│   │   ├── recommendation_service.py  # Course recommendation engine
│   │   ├── rag_service.py        # RAG pipeline (hybrid retrieval + LLM)
│   │   ├── agentic_service.py    # LangGraph agent orchestration
│   │   └── recruitment_service.py  # Resume-JD matching agent
│   ├── validation/
│   │   └── schemas.py            # Pydantic request/response models
│   └── utils/                    # Utility functions
├── data/                         # Data directory (contents gitignored)
│   ├── raw/                      # Raw input datasets
│   ├── processed/                # Cleaned and transformed data
│   ├── hr_policies/              # HR policy PDFs for RAG
│   ├── resumes/                  # Sample resumes for recruitment agent
│   ├── job_descriptions/         # Sample JDs for recruitment agent
│   ├── predictions/              # Runtime prediction logs
│   └── chroma_db/                # Chroma vector database (gitignored)
├── notebooks/                    # Jupyter notebooks (exploration & analysis)
│   └── figures/                  # Generated SHAP plots (gitignored)
├── models/                       # Trained model artifacts
│   └── v1/                       # Version 1 model (binary gitignored)
├── tests/                        # Pytest test suite
├── frontend/
│   └── app.py                    # Streamlit dashboard
├── docs/
│   └── data_relationships.md     # Data schema documentation
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/enterprise_hr_ai.git
   cd enterprise_hr_ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key for RAG LLM generation
   ```

5. **Place your data files:**
   - Place raw CSV files in `data/raw/` (see `01_PROJECT_ROADMAP.md` for the expected file mapping)
   - Run the numbered notebooks in `notebooks/` to generate processed data and train models

### Running the Application

**Start the FastAPI backend:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Start the Streamlit dashboard (in a separate terminal):**
```bash
streamlit run frontend/app.py
```

The dashboard will be available at `http://localhost:8501` and the API at `http://localhost:8000`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/attrition` | POST | Predict attrition risk for an employee |
| `/dashboard/summary` | GET | Get dashboard KPI summary |
| `/dashboard/attrition-by-department` | GET | Get attrition risk by department |
| `/dashboard/skill-gaps` | GET | Get organizational skill gaps |
| `/dashboard/recommendations` | GET | Get course recommendations |
| `/employees/{employee_id}` | GET | Get single employee intelligence |
| `/rag/ask` | POST | Ask a question about HR policies |
| `/agent/workforce-check` | POST | Run agentic workforce intelligence check |

### Running Tests

```bash
pytest
```

## Architecture

The system follows a layered architecture:

1. **Data Layer** — Raw HR datasets, O*NET occupation/skill taxonomies
2. **ML Layer** — Attrition prediction model with SHAP explainability
3. **Intelligence Layer** — Skill gap engine, engagement analytics, recommendations
4. **Application Layer** — FastAPI backend, Pydantic validation, service modules
5. **Knowledge Layer** — RAG pipeline with hybrid retrieval and LLM grounding
6. **Agentic Layer** — LangGraph orchestrator with governed tool-calling
7. **Presentation Layer** — Streamlit dashboard

## Known Limitations

- **RAG latency**: ~5 seconds per query due to free-tier LLM API. For production, use a paid model or cache results.
- **Synthetic data**: `employee_skills.csv` and `courses.csv` are synthesized (clearly documented in code). Replace with real data for production use.
- **Course catalog**: Limited to 35 synthetic courses. Expand for broader skill coverage.
- **Agent routing**: Uses simplified rule-based routing, not full LLM-driven tool selection.

## License

This project is for demonstration and educational purposes. Sample data files are from public datasets (Kaggle, O*NET).
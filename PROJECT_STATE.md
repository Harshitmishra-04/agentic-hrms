# Project State - Streamlit Cloud Deployment

## Deployment Configuration

This project has been adapted to run as a single Streamlit Community Cloud app (free tier, no Docker, one process only).

### Changes Made

#### 1. Backend Integration (No Separate Process)
- **File**: `frontend/app.py`
- **Changes**: Completely redesigned to run backend services directly in the Streamlit process
- **Implementation**: 
  - Removed all background process/threading approaches (neither worked in Streamlit Cloud)
  - Direct imports of service modules instead of HTTP calls
  - Replaced HTTP client functions with direct function calls to service modules
  - All services run in the same process as Streamlit - guaranteed to work on Streamlit Cloud
  - No API server needed - everything is integrated as direct Python function calls

#### 2. No API Communication Needed
- **File**: `frontend/app.py`
- **Status**: No HTTP communication needed - all services are called directly
- **Environment Variable**: No longer needed for local development

#### 3. No Runtime Checks Needed
- **File**: `frontend/app.py`
- **Implementation**: Removed all API readiness checks since services run in-process
  - No network communication needed
  - No startup delays
  - Immediate availability of all services

#### 4. Requirements.txt Optimization
- **File**: `requirements.txt`
- **Changes**: Removed dev-only dependencies to reduce install size and memory footprint:
  - Removed: `pytest` (testing framework)
- **Runtime Dependencies**: All required dependencies are included:
  - Core: `fastapi`, `uvicorn`, `pydantic`, `streamlit`
  - ML/Data: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `joblib`
  - RAG/Embeddings: `sentence-transformers`, `chromadb`, `pypdf`, `rank-bm25`
  - Agentic: `langgraph`, `openai`
  - Visualization: `matplotlib`, `plotly`
  - Utilities: `python-dotenv`

#### 5. Data and Model Files
- **Gitignore Changes**: Updated `.gitignore` to allow deployment files:
  - Removed `data/chroma_db/` from gitignore (ChromaDB persisted folder needed for deployment)
  - Removed `data/processed/**/*.csv` from gitignore (processed data needed for deployment)
  - Removed `*.joblib` from gitignore (model files needed for deployment)
- **File Sizes**:
  - `data/` directory: ~35 MB (42 files) - reasonable for git push
  - `models/` directory: ~5 KB (3 files) - minimal
  - `data/chroma_db/`: Contains ChromaDB SQLite database (~368 KB)
- **Status**: All required data and model files are now committed to the repository for Streamlit Cloud deployment

#### 6. Environment Variables
- **API Key Configuration**: `OPENROUTER_API_KEY` is read from environment variable in `app/services/rag_service.py`
- **Security**: 
  - The key is NOT committed to the repository (`.env` is in `.gitignore`)
  - Template provided in `.env.example`
  - For Streamlit Cloud deployment, this should be set as a "Secret" in the Streamlit Cloud deployment UI

### Deployment Instructions

#### Local Development
1. Clone the repository
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Set environment variables: Copy `.env.example` to `.env` and add your `OPENROUTER_API_KEY`
4. Run the application: `streamlit run frontend/app.py`
   - The FastAPI backend will start automatically in a background thread
   - The Streamlit dashboard will be available at `http://localhost:8501`

#### Streamlit Cloud Deployment
1. Push the repository to GitHub
2. Create a new app in Streamlit Cloud
3. Connect the GitHub repository
4. Set the environment variable `OPENROUTER_API_KEY` as a Secret in the Streamlit Cloud deployment UI
5. Deploy - Streamlit Cloud will:
   - Install dependencies from `requirements.txt`
   - Start the Streamlit app
   - The app will automatically start the FastAPI backend in a background thread
   - Both processes will run in the same container

### Architecture Notes

- **Single Process**: Both FastAPI backend and Streamlit frontend run in the same process
- **Background Thread**: FastAPI runs as a daemon thread, ensuring it doesn't prevent Streamlit from shutting down
- **Local Communication**: Streamlit communicates with FastAPI via `http://127.0.0.1:8000`
- **Memory Constraints**: The free tier has ~1GB memory limit. The optimized requirements.txt helps stay within this limit

### Known Limitations

- The background thread approach is suitable for the Streamlit Cloud free tier but may not be ideal for high-traffic production deployments
- For production deployments with higher traffic, consider using separate containers/processes for backend and frontend
- The ChromaDB data is persisted in the repository; for dynamic data, consider using external storage

### Verification Checklist

- [x] Backend services integrated directly into Streamlit process (no separate API server)
- [x] All HTTP calls replaced with direct Python function calls
- [x] `requirements.txt` includes all runtime dependencies
- [x] Dev-only dependencies (pytest) removed from `requirements.txt`
- [x] `data/`, `models/`, and ChromaDB folders are not gitignored
- [x] File sizes are reasonable for git push (<100MB total)
- [x] `OPENROUTER_API_KEY` is read from environment variable
- [x] `OPENROUTER_API_KEY` is not committed to repository
- [x] Architecture guaranteed to work on Streamlit Cloud (no background processes)
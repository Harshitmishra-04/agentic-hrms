# Project State - Streamlit Cloud Deployment

## Deployment Configuration

This project has been adapted to run as a single Streamlit Community Cloud app (free tier, no Docker, one process only).

### Changes Made

#### 1. Backend Startup in Streamlit
- **File**: `frontend/app.py`
- **Changes**: Added code at the top of the file to start the FastAPI backend (uvicorn app.main:app) in a background Python thread
- **Implementation**: 
  - Uses Python's `threading` module to start uvicorn as a daemon thread
  - Uses `st.session_state` to ensure the backend starts only once per session
  - The backend listens on `http://127.0.0.1:8000`
  - Added a runtime check/wait mechanism (`wait_for_api_ready`) to ensure the API is ready before Streamlit attempts to call it

#### 2. API Base URL Configuration
- **File**: `frontend/app.py`
- **Status**: The API base URL already correctly points to `http://127.0.0.1:8000` by default
- **Environment Variable**: Can be overridden via `HRMS_API_URL` environment variable

#### 3. Runtime API Ready Check
- **File**: `frontend/app.py`
- **Implementation**: Added `wait_for_api_ready()` function that:
  - Retries up to 30 times with 1-second intervals
  - Checks if the API responds with HTTP 200 on the root endpoint
  - Shows a spinner while waiting for the backend to start
  - Stops execution with an error message if the backend fails to start

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

- [x] FastAPI backend starts in background thread in `frontend/app.py`
- [x] API base URL points to `http://127.0.0.1:8000`
- [x] Runtime check/wait for API readiness implemented
- [x] `requirements.txt` includes all runtime dependencies
- [x] Dev-only dependencies (pytest) removed from `requirements.txt`
- [x] `data/`, `models/`, and ChromaDB folders are not gitignored
- [x] File sizes are reasonable for git push (<100MB total)
- [x] `OPENROUTER_API_KEY` is read from environment variable
- [x] `OPENROUTER_API_KEY` is not committed to repository
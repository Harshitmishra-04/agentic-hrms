from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import router
from app.utils.logging_config import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger = get_logger()
    logger.info("Agentic HRMS API starting up")
    yield
    logger.info("Agentic HRMS API shutting down")


app = FastAPI(
    title="Agentic HRMS Workforce Intelligence API",
    description="Backend API serving attrition predictions, employee intelligence profile lookup, and upskilling course recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Map Pydantic/FastAPI validation failures to HTTP 400 before business logic runs."""
    logger = get_logger()
    logger.warning(
        "Validation failed for %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Agentic HRMS backend API",
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

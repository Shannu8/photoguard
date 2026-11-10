"""Main FastAPI application entry point for Mathematical Image Analysis System."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("photoguard.main")

app = FastAPI(
    title="Mathematical Image Analysis and AI Protection System (Phase 1 & 2)",
    description=(
        "Educational and university-level research application for mathematical matrix analysis, "
        "eigenvalues, Singular Value Decomposition (SVD), and low-rank image reconstruction."
    ),
    version="1.0.0",
)

# Explicit origins to adhere strictly to browser CORS policy with credentials
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "Mathematical Image Analysis System",
        "phases_active": ["Phase 1: Image Processing Foundation", "Phase 2: Mathematical Image Analysis"],
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PhotoGuard FastAPI server on 127.0.0.1:8000...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

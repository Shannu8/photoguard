"""Main FastAPI application entry point for Mathematical Image Analysis System."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Mathematical Image Analysis and AI Protection System (Phase 1 & 2)",
    description=(
        "Educational and university-level research application for mathematical matrix analysis, "
        "eigenvalues, Singular Value Decomposition (SVD), and low-rank image reconstruction."
    ),
    version="1.0.0",
)

# Configure CORS for local development with Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "Mathematical Image Analysis System",
        "phases_active": ["Phase 1: Image Processing Foundation", "Phase 2: Mathematical Image Analysis"],
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


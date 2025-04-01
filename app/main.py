from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.routes.api import router as api_router

# Get LLM provider from environment variables
llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

# Create FastAPI app
app = FastAPI(
    title="Car Issue Classification API",
    description="API for classifying car issues into groups and categories",
    version="1.0.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """
    Root endpoint that returns info about the API.
    """
    return {
        "message": "Car Issue Classification API",
        "version": "1.0.0",
        "llm_provider": llm_provider.capitalize(),
        "documentation": "/docs",
    } 
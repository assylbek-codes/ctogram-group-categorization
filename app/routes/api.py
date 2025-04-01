from fastapi import APIRouter, HTTPException
from app.models.schema import CarIssueRequest, CarIssueResponse
from app.utils.llm import (
    classify_car_issue_with_hashtags
)

router = APIRouter()


@router.post("/classify", response_model=CarIssueResponse)
async def classify_issue(request: CarIssueRequest):
    """
    Classify a car issue text into a group and categories.
    Uses a LLM classification.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        group, group_id, categories_ids, categories = classify_car_issue_with_hashtags(request.text)
        
        return CarIssueResponse(
            group=group,
            group_id=group_id,
            categories_ids=categories_ids,
            categories=categories,
            method_used="llm"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying car issue: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "healthy"} 
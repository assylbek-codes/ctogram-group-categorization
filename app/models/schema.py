from pydantic import BaseModel
from typing import List, Dict, Optional


class CarIssueRequest(BaseModel):
    text: str

class CarIssueResponse(BaseModel):
    group: str
    group_id: int
    categories: Dict[int, str]
    categories_ids: List[int]
    method_used: Optional[str] = None 
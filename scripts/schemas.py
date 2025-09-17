from pydantic import BaseModel
from typing import Optional, List

class LabelFilter(BaseModel):
    ns: str
    code: Optional[str] = None
    prefix: Optional[str] = None

class Question(BaseModel):
    text: str
    top_k: int = 8
    search_methods: Optional[List[str]] = None
    filters: Optional[List[LabelFilter]] = None
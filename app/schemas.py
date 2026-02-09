from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class RunRequest(BaseModel):
    task: str = Field(..., min_length=1)
    allow_web_search: bool = False

class ToolLog(BaseModel):
    name: str
    arguments: Dict[str, Any]
    output: Any

class RunResponse(BaseModel):
    model_used: str
    final_text: str
    tool_logs: List[ToolLog] = []

class CreateVectorStoreRequest(BaseModel):
    name: str = Field(default="My Vector Store")

class CreateVectorStoreResponse(BaseModel):
    vector_store_id: str
    name: str

class UploadToVectorStoreResponse(BaseModel):
    vector_store_id: str
    file_ids: List[str]

class AskFileSearchRequest(BaseModel):
    vector_store_id: str
    question: str
    model: Optional[str] = None

class AskFileSearchResponse(BaseModel):
    model_used: str
    answer: str
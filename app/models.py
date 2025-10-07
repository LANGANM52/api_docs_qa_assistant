from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentUpload(BaseModel):
    """Model for uploading API documentation."""
    content: str = Field(..., description="The API documentation content")
    metadata: Optional[dict] = Field(default={}, description="Additional metadata")
    doc_id: Optional[str] = Field(default=None, description="Unique document identifier")


class QuestionRequest(BaseModel):
    """Model for asking questions about API documentation."""
    question: str = Field(..., min_length=5, description="The question to ask")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens in response")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")


class Source(BaseModel):
    """Model for source document information."""
    content: str
    metadata: dict
    relevance_score: float


class Answer(BaseModel):
    """Model for the response to a question."""
    model_config = {"protected_namespaces": ()}

    question: str
    answer: str
    sources: List[Source]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: str
    tokens_used: Optional[int] = None


class HealthCheck(BaseModel):
    """Model for health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    vector_store_status: str
    llm_status: str


class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
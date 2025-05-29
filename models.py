# This file contains all Pydantic models for data validation and structure
# Used for enforcing data types and validation across the application

from pydantic import BaseModel, Field
from typing import List, Optional, Any

# Work Experience Models
class WorkExperience(BaseModel):
    company: str
    role: str
    start_date: Optional[str] = Field(default=None, description="YYYY-MM")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM or Present")
    description: str

class WorkExperienceList(BaseModel):
    work_experiences: List[WorkExperience]

# Education Models
class Education(BaseModel):
    institution: str
    degree: str
    field: str
    start_year: Optional[int] = Field(default=None, description="Start year of education, if available") 
    end_year: Optional[int] = Field(default=None, description="End year of education, if available")

class EducationList(BaseModel):
    education: List[Education]

# API Request/Response Models
class QuestionRequest(BaseModel):
    question: str

class SummaryResponse(BaseModel):
    summary: str
    success: bool
    error: Optional[str] = None

class InsightsResponse(BaseModel):
    insights: List[str]

class InterviewQuestionsResponse(BaseModel):
    questions: List[str]

# File Upload Response Model
class FileUploadResponse(BaseModel):
    message: str
    file_id: str
    original_name: str
    chunks_count: int

# For Stored Files List
class StoredFileMeta(BaseModel):
    original_name: str
    upload_date: str
    expiry_date: str
    vector_store_path: str

class StoredFilesListResponse(BaseModel):
    files: dict[str, StoredFileMeta]

# For streaming events if using Server-Sent Events (SSE)
class StreamEvent(BaseModel):
    type: str 
    content: Any

# --- Part C: Model for Resuming Question Generation ---
class ResumeQuestionLangGraphRequest(BaseModel):
    checkpoint_id: str  # This is the thread_id from a previous LangGraph run
    resume_summary: Optional[str] = Field(default=None)
    resume_insights: Optional[List[str]] = Field(default=None)
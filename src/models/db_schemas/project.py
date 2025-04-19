from pydantic import BaseModel, Field, validator, constr, conint, EmailStr, HttpUrl
from typing import Optional as Optinal, List, Dict, Any
from bson import ObjectId
class Project(BaseModel):
    _id: Optinal[str]
    project_id: Optinal[str] = Field(..., min_length=1)

    @validator("project_id")
    def validate_project_id(cls, value):
        if not value.isalnum():
            raise ValueError("Project ID cannot be alphanumeric")
        return value

    class Config:
        arbitrary_types_allowed = True

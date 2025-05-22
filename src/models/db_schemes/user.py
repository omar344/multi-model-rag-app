from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from bson import ObjectId

class User(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    username: str = Field(..., min_length=3)
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False

    model_config = {
        "arbitrary_types_allowed": True
    }
from typing import Optional, Dict, Union
from pydantic import BaseModel, Field
from bson import ObjectId
from models.enums.ProcessingEnum import ProcessingEnum

class DataChunk(BaseModel):
    _id: Optional[ObjectId]
    
    chunk_text: Optional[str] = Field(None, min_length=1)
    chunk_image: Optional[str] = Field(None, min_length=1)  
    chunk_table: Optional[Dict] = None  
    chunk_metadata: Dict  
    user_id: ObjectId
    chunk_order: int = Field(..., gt=0)  
    chunk_project_id: ObjectId
    file_type: str = Field(..., min_length=1)

    model_config = {
        "arbitrary_types_allowed": True
    }

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [
                    ("chunk_project_id", 1),
                    ("user_id", 1)  # <-- Add user_id to index
                ],
                "name": "chunk_project_id_user_id_index_1",
                "unique": False
            }
        ]

class RetrievedDocument(BaseModel):
    text: str
    score: float
    metadata: dict = {}
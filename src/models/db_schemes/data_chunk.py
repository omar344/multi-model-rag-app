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
    
    chunk_order: int = Field(..., gt=0)  
    chunk_project_id: ObjectId
    file_type: ProcessingEnum

    class Config:
        arbitrary_types_allowed = True  # Allows usage of non-Pydantic types like ObjectId

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [
                    ("chunk_project_id", 1)
                ],
                "name": "chunk_project_id_index_1",
                "unique": False
            }
        ]
    
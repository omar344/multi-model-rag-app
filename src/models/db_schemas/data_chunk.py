from pydantic import BaseModel, Field
from bsan.objectid import ObjectId

class DataChunk(BaseModel):
    _id: Optinal[ObjectId]
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id: str = ObjectId

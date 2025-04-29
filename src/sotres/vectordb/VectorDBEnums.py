from enum import Enum

class VectorDBEnum(Enum):
    QDRANT: "QDRANT"
    
class DistanceMethodEnum(Enum):
    COSINE = "cosine"
    DOT = "dot"
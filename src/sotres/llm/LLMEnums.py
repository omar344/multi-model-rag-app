from enum import Enum

class LLMEnum(Enum):
    GROQ = "GROQ"
    HUGGINGFACE = "HUGGINGFACE"
    
    
class GroqEnum(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    

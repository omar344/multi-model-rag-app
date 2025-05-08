from enum import Enum

class LLMEnums(Enum):
    GROQ = "GROQ"
    HUGGINGFACE = "HUGGINGFACE"
    OVHCLOUD = "OVHCLOUD"
    OPENAI = "OPENAI"
    AZURE_OPENAI = "AZURE_OPENAI"
    VOYAGEAI = "VOYAGEAI"
    
    
class GroqEnum(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class OVHCloudEnum(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class OpenAIEnum(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class HuggingFaceEnum(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class AzureOpenAIEnum(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"
    IMAGE = "image"   

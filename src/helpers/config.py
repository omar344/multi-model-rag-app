from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: str
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int 

    MONGODB_URL: str
    MONGODB_DATABASE: str
    
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str
    
    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    OPENAI_API_VERSION: str = None
    AZURE_OPENAI_API_KEY: str = None
    AZURE_OPENAI_ENDPOINT: str = None
    AZURE_OPENAI_API_VERSION: str = None
    GROQ_API_KEY: str = None
    HuggingFace_API_KEY: str = None
    OVHCloud_API_KEY: str = None
    GITHUB_TOKEN: str = None
    VOYAGE_AI_KEY: str = None
    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None

    INPUT_DEFAULT_MAX_CHARACTERS: int = None
    GENERATION_DEFAULT_MAX_TOKENS: int = None
    GENERATION_DEFAULT_TEMPERATURE: float = None
    
    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None
    
    PRIMARY_LANGUAGE: str = "en"
    DEFAULT_LANGUAGE: str = "en"

    class Config:
        env_file = '.env'

def get_settings():
    return Settings()

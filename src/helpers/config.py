from pydantic_setting import BaseSettings, SettingsConfigDict

class config(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str

    class Config:
        env_file = '.env'
        

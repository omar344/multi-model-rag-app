from .LLMEnums import LLMEnums
from .providers import HuggingFaceProvider, OpenAIProvider, GroqProvider

class LLMFactory:
    def __init__(self, config: dict):
        self.config = config
        
    def create(self, provider: str):
        if provider == LLMEnums.HUGGINGFACE.value:
            return HuggingFaceProvider(
                api_key=self.config.HuggingFace_API_KEY,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )
            
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                api_url = self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )
        
        if provider == LLMEnums.GROQ.value:
            return GroqProvider(
                api_key=slef.config.GROQ_API_KEY,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )
        
        return None
        
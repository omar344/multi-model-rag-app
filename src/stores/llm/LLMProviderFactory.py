from .LLMEnums import LLMEnums
from .providers import HuggingFaceProvider, OpenAIProvider, GroqProvider, AzureOpenAIProvider, VoyageAIProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config
        
    def create(self, provider: str):
        if provider == LLMEnums.HUGGINGFACE.value:
            return HuggingFaceProvider(
                api_key=self.config.HuggingFace_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                api_url = self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )
        if provider == LLMEnums.AZURE_OPENAI.value:
            return AzureOpenAIProvider(
                api_key=self.config.AZURE_OPENAI_API_KEY,
                endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                api_version=self.config.AZURE_OPENAI_API_VERSION,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )
        if provider == LLMEnums.GROQ.value:
            return GroqProvider(
                api_key=self.config.GROQ_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )
        if provider == LLMEnums.VOYAGEAI.value:
            return VoyageAIProvider(
                api_key=self.config.VOYAGE_AI_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )
        return None

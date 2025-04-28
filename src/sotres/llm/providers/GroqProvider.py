from ..LLMInterface import LLMInterface
from ..LLMEnums import LLMEnums

from groq import Groq

class GroqProvider(LLMInterface):
    def __init__(self, api_key: str, api_url: str=None,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature
        
        self.generation_model_id = None
        
        self.embedding_model_id = None
        self.embedding_size = None
        
        self.client = Groq(api_key=self.api_key)
        
        self.logger = logging.getLogger(__name__)

        
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        if not self.client:
            raise ValueError("Groq client is not initialized.")
            return None
        
        if not self.generation_model_id:
            raise ValueError("Generation model is not set.")
            return None 
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        
        chat_history.append(
            self.construct_prompt(prompt=prompt, role=GroqEnum.USER.value)
        )
        
        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=chat_history,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
        
        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("No response from Groq API.")
            return None
        
        return response.choices[0].message["content"]
    
    def embed_text(self, text: str):
        raise NotImplementedError("Embedding is not implemented for Groq API.")
    
    def embed_image(self, image_base64: str):
        raise NotImplementedError("Embedding is not implemented for Groq API.")
    
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
from abc import abstractmethod
from ..LLMInterface import LLMInterface
from ..LLMEnums import LLMEnums

from sentence_transformers import SentenceTransformer
import base64
import logging
from io import BytesIO
from PIL import Image


class HuggingFaceProvider(LLMInterface):
    
    def __init__(self, api_key: str, api_url: str=None,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        self.model_id = model_id
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = model_id
        self.embedding_size = None
        
        self.client = SentenceTransformer(model_id)
        
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        raise NotImplementedError("HuggingFaceProvider does not support text generation.")
    
    def embed_text(self, text: str):
        if not self.client:
            raise ValueError("HuggingFace client is not initialized.")
        
        if not self.embedding_model_id:
            raise ValueError("Embedding model is not set.")
        
        text = self.process_text(text)
        return self.client.encode(text, convert_to_numpy=True)
    
    
    def base64_to_binary(self, base64_string: str):
        return base64.b64decode(base64_string)
    
    def embed_image(self, image_base64: str):
        if not self.client:
            raise ValueError("HuggingFace client is not initialized.")
        
        if not self.embedding_model_id:
            raise ValueError("Embedding model is not set.")
        
        image_data = self.base64_to_binary(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        embeddings = self.client.encode(image, convert_to_numpy=True)
        if embeddings is None:
            raise ValueError("Failed to embed image.")
        
        return embeddings

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
        


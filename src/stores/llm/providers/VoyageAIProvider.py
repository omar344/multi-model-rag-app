from ..LLMInterface import LLMInterface
from ..LLMEnums import LLMEnums
import voyageai
import logging
import base64
from io import BytesIO
from PIL import Image
import time
import random

class VoyageAIProvider(LLMInterface):
    
    def __init__(self, api_key: str, api_url: str=None,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = voyageai.Client(
            api_key=self.api_key
        )

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        pass

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        pass

    def _base64_to_pil(self, base64_str):
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            self.logger.error(f"Failed to decode base64 image: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        """
        If document_type == "document" or "query", treat as text.
        If document_type == "image", treat text as base64 image.
        """
        if not self.client:
            self.logger.error("VoyageAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for VoyageAI was not set")
            return None

        input_type = None
        if document_type == "document":
            input_type = "document"
        elif document_type == "query":
            input_type = "query"

        # Handle image (base64) or text
        if document_type == "image":
            pil_img = self._base64_to_pil(text)
            if pil_img is None:
                return None
            inputs = [[pil_img]]
        else:
            # Just text
            inputs = [[text]]

        # Retry logic for connection errors
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                result = self.client.multimodal_embed(
                    inputs=inputs,
                    model=self.embedding_model_id,
                    input_type=input_type,
                    truncation=True
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                if any(conn_error in error_str for conn_error in ["connection", "remote", "timeout", "aborted", "network"]):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Connection error on attempt {attempt + 1}: {e}. Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"Connection error after {max_retries} attempts: {e}")
                        return None
                else:
                    self.logger.error(f"Error while embedding with VoyageAI: {e}")
                    return None

        if result and result.embeddings and len(result.embeddings) > 0:
            return result.embeddings[0]
        else:
            self.logger.error("No embedding returned from VoyageAI multimodal_embed")
            return None

    def construct_prompt(self, prompt: str, role: str):
        pass

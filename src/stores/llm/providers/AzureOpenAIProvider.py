from ..LLMInterface import LLMInterface
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import logging

class AzureOpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, endpoint: str, api_version: str,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
            # credential=AzureKeyCredential(self.api_key)
        )
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
        if not self.client:
            self.logger.error("AzureOpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for AzureOpenAI was not set")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append({
            "role": "user",
            "content": self.process_text(prompt)
        })

        response = self.client.chat.completions.create(
            model=self.generation_model_id,  # deployment name
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with AzureOpenAI")
            return None

        return response.choices[0].message["content"]

    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("AzureOpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for AzureOpenAI was not set")
            return None

        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model_id  # deployment name
        )

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with AzureOpenAI")
            return None

        return response.data[0].embedding

    def embed_image(self, image_base64: str):
        raise NotImplementedError("Image embedding not implemented for AzureOpenAI.")

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
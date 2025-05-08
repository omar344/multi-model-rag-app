from abc import ABC, abstractmethod
from ..LLMInterface import LLMInterface
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
import os
import logging


class GitHubModelsProvider(LLMInterface):
    def __init__(self, api_key: str = None, api_url: str = None,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        self.api_key = api_key
        self.endpoint = api_url
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = EmbeddingsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        raise NotImplementedError("Text generation is not supported for GitHubModelsProvider.")

    def _ensure_data_url(self, image_b64: str, mime_type: str = "image/jpeg") -> str:
        """Ensure the base64 string is a data URL. If already a data URL, return as is."""
        if image_b64.startswith("data:"):
            return image_b64
        return f"data:{mime_type};base64,{image_b64}"

    def embed_text(self, text: str, document_type: str = None):
        """
        Embeds a text or a base64-encoded image (as data URL) using Cohere's Embed v4 model via Azure AI Inference.
        Accepts either a string (text or image data URL) or a list of such strings.
        """
        if not self.embedding_model_id:
            self.logger.error("Embedding model is not set.")
            return None

        # Handle image input
        if document_type == "image":
            if isinstance(text, str):
                text = self._ensure_data_url(text)
            elif isinstance(text, list):
                text = [self._ensure_data_url(t) for t in text]
            else:
                self.logger.error("Input to embed_text must be a string or list of strings.")
                return None

        # Accept both single string and list input
        if isinstance(text, str):
            inputs = [text]
        elif isinstance(text, list):
            inputs = text
        else:
            self.logger.error("Input to embed_text must be a string or list of strings.")
            return None

        try:
            response = self.client.embed(
                input=inputs,
                model=self.embedding_model_id
            )
            # Return the embedding for the first input (for single input)
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                self.logger.error("No embedding returned from GitHub Models API.")
                return None
        except Exception as e:
            self.logger.error(f"Error while embedding with GitHub Models API: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt
        }
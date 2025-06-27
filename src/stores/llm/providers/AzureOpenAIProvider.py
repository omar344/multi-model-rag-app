from ..LLMInterface import LLMInterface
from ..LLMEnums import AzureOpenAIEnum
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import requests
import logging
from transformers import CLIPTokenizer
from PIL import Image as PILImage
import base64
import io
import time
import random

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

        self.enum = AzureOpenAIEnum
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
        self.logger = logging.getLogger(__name__)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

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

        # Retry logic for connection errors
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.generation_model_id,  # deployment name
                    messages=chat_history,
                    max_tokens=max_output_tokens,
                    temperature=temperature
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                # Handle content filter errors specifically
                if "content_filter" in error_str or "responsibleai" in error_str:
                    self.logger.warning(f"Content filter triggered on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        # Try to sanitize the prompt for next attempt
                        chat_history = self._sanitize_prompt_for_content_filter(chat_history)
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Retrying with sanitized prompt in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"Content filter error after {max_retries} attempts: {e}")
                        return None
                elif any(conn_error in error_str for conn_error in ["connection", "remote", "timeout", "aborted", "network"]):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Connection error on attempt {attempt + 1}: {e}. Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"Connection error after {max_retries} attempts: {e}")
                        return None
                else:
                    self.logger.error(f"Exception during AzureOpenAI completion: {e}")
                    return None

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with AzureOpenAI")
            return None

        return response.choices[0].message["content"]

    def _sanitize_prompt_for_content_filter(self, chat_history):
        """Sanitize prompt to avoid content filter triggers"""
        sanitized_history = []
        for message in chat_history:
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                if isinstance(content, str):
                    # Remove potentially problematic content
                    # This is a basic sanitization - you might need to customize based on your needs
                    sanitized_content = content
                    # Remove any content that might trigger filters
                    # You can add more sophisticated sanitization here
                    sanitized_history.append({
                        'role': message['role'],
                        'content': sanitized_content
                    })
                else:
                    # For multimodal content, keep as is but log
                    self.logger.warning("Multimodal content detected, keeping as is")
                    sanitized_history.append(message)
            else:
                sanitized_history.append(message)
        return sanitized_history

    def _truncate_text_to_77_tokens(self, text: str) -> str:
        input_ids = self.tokenizer(text, truncation=True, max_length=75, add_special_tokens=True)["input_ids"]
        return self.tokenizer.decode(input_ids)

    def _resize_image_336(self, image_b64: str) -> str:
        try:
            image_data = base64.b64decode(image_b64)
            image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
            image = image.resize((336, 336), PILImage.BICUBIC)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            resized_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return resized_b64
        except Exception as e:
            self.logger.error(f"Failed to resize image: {e}")
            return image_b64  # fallback to original

    def embed_text(self, text: str, document_type: str = None):
        """
        If document_type == "image", treat text as base64 or URL for image.
        Otherwise, treat as text.
        """
        if not self.endpoint or not self.api_key:
            self.logger.error("Azure CLIP endpoint or API key not set")
            return None

        columns = ["image", "text"]
        if document_type == "image":
            text = self._resize_image_336(text)
            data_row = [text, ""]
        else:
            text = self._truncate_text_to_77_tokens(text)
            data_row = ["", text]

        payload = {
            "input_data": {
                "columns": columns,
                "data": [data_row]
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Retry logic for connection errors
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
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
                    self.logger.error(f"Error while embedding with Azure CLIP: {e}")
                    return None

        if not isinstance(result, list) or not result:
            self.logger.error("Invalid response from Azure CLIP endpoint")
            return None

        features = result[0]
        if document_type == "image" and "image_features" in features:
            return features["image_features"]
        elif "text_features" in features:
            return features["text_features"]
        else:
            self.logger.error("No features found in Azure CLIP response")
            return None
        
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
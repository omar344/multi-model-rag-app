from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnum
from openai import OpenAI
import logging
import time
import random

class OpenAIProvider(LLMInterface):

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
        
        self.enum = OpenAIEnum

        self.client = OpenAI(
            api_key = self.api_key,
            base_url = self.api_url
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
        self.logger.info("Starting OpenAI text generation.")
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        self.logger.debug(f"Appending user prompt to chat history. Prompt type: {type(prompt)}")
        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnum.USER.value)
        )

        self.logger.info(f"Sending request to OpenAI: model={self.generation_model_id}, max_tokens={max_output_tokens}, temperature={temperature}")
        
        # Retry logic for connection errors
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.generation_model_id,
                    messages=chat_history,
                    max_tokens=max_output_tokens,
                    temperature=temperature
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
                    self.logger.error(f"Exception during OpenAI completion: {e}")
                    return None

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        self.logger.info("Received response from OpenAI.")
        return response.choices[0].message.content

    def embed_text(self, text: str, document_type: str = None):
        self.logger.info("Starting OpenAI embedding.")
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return 

        max_size = self.embedding_size or 1536
        text = text[:max_size]
        self.logger.debug(f"Embedding text of length {len(text)} with model {self.embedding_model_id}")

        # Retry logic for connection errors
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model_id,
                    input=text,
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
                    self.logger.error(f"Exception during OpenAI embedding: {e}")
                    return None

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None

        self.logger.info("Received embedding from OpenAI.")
        return response.data[0].embedding

    def construct_prompt(self, prompt: str, role: str):
        self.logger.debug(f"Constructing prompt for role: {role}, prompt type: {type(prompt)}")
        if isinstance(prompt, list):
            # If the prompt is a list, use it directly as the content
            self.logger.info("Prompt is a list (multimodal). Using as content.")
            return {
                "role": role,
                "content": prompt
            }
        # Fallback for text-only prompts
        self.logger.info("Prompt is text. Processing as text-only content.")
        return {
            "role": role,
            "content": self.process_text(prompt)
        }





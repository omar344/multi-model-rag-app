import logging
from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
import json
from typing import List

class NLPController(BaseController):
    
    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: str, user_id: str):
        return f"collection_{user_id}_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id, user_id=project.user_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id, user_id=project.user_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        logger = logging.getLogger("uvicorn.error")
        if not chunks:
            logger.warning("No chunks to embed for this batch.")
            return 0

        collection_name = self.create_collection_name(project_id=project.project_id, user_id=project.user_id)
        logger.info(f"Indexing into vector DB collection: {collection_name}")

        texts = []
        metadata = []
        vectors = []
        valid_ids = []

        for i, c in enumerate(chunks):
            chunk_type = c.chunk_metadata.get("type")
            logger.debug(f"Processing chunk {i} of type {chunk_type}")
            if chunk_type == "image" and c.chunk_text:
                logger.info(f"Embedding image chunk at index {i}")
                vector = self.embedding_client.embed_text(
                    text=c.chunk_text, document_type=DocumentTypeEnum.IMAGE.value
                )
                text_val = c.chunk_text
            else:
                logger.info(f"Embedding text chunk at index {i}")
                vector = self.embedding_client.embed_text(
                    text=c.chunk_text, document_type=DocumentTypeEnum.DOCUMENT.value
                )
                text_val = c.chunk_text

            if vector is not None and len(vector) == self.embedding_client.embedding_size:
                vectors.append(vector)
                texts.append(text_val)
                metadata.append(c.chunk_metadata)
                valid_ids.append(chunks_ids[i])
                logger.debug(f"Chunk {i} embedded and added to batch.")
            else:
                logger.warning(f"Skipping vector at index {i}: None or wrong size ({None if vector is None else len(vector)})")

        logger.info(f"Creating collection '{collection_name}' with embedding size {self.embedding_client.embedding_size}")
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        logger.info(f"Inserting {len(vectors)} vectors into collection '{collection_name}'")
        inserted = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=valid_ids,
        )

        inserted_count = len(vectors) if inserted else 0
        logger.info(f"Inserted {inserted_count} vectors into collection '{collection_name}'.")

        return True

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
        logger = logging.getLogger("uvicorn.error")
        logger.info(f"Searching vector DB for project {project.project_id} with query: {text}")

        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id, user_id=project.user_id)
        logger.debug(f"Using collection: {collection_name}")

        # step2: get text embedding vector
        vector = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)
        logger.debug(f"Query embedding vector generated.")

        if not vector or len(vector) == 0:
            logger.warning("No embedding vector generated for query.")
            return False

        # step3: do semantic search
        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        if not results:
            logger.warning("No results found in vector DB search.")
            return False

        logger.info(f"Found {len(results)} results in vector DB search.")
        return results
    
    def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        logger = logging.getLogger("uvicorn.error")
        logger.info(f"Starting RAG answer for project {project.project_id} with query: {query}")
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )
        if not retrieved_documents or len(retrieved_documents) == 0:
            logger.warning("No documents retrieved from vector DB.")
            return answer, full_prompt, chat_history

        logger.info(f"Retrieved {len(retrieved_documents)} documents from vector DB.")

        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")
        message_parts = []

        for idx, doc in enumerate(retrieved_documents):
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "image":
                logger.info(f"Adding image document {idx} to prompt.")
                image_b64 = doc.text
                # If not already a data URL, wrap it
                if not image_b64.startswith("data:"):
                    image_b64 = f"data:image/png;base64,{image_b64}"
                message_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_b64}
                })
            else:
                logger.info(f"Adding text document {idx} to prompt.")
                # Add text document
                message_parts.append({
                    "type": "text",
                    "text": self.template_parser.get("rag", "document_prompt", {
                        "doc_num": idx + 1,
                        "chunk_text": doc.text,
                    })
                })

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})
        message_parts.append({"type": "text", "text": footer_prompt})
        logger.info("Prompt for LLM constructed.")

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enum.SYSTEM.value,
            )
        ]
        logger.info("Chat history for LLM constructed.")

        # step4: Retrieve the Answer
        logger.info("Sending prompt to generation client.")
        answer = self.generation_client.generate_text(
            prompt=message_parts,
            chat_history=chat_history
        )

        if answer:
            logger.info("Received answer from generation client.")
        else:
            logger.warning("No answer received from generation client.")

        return answer, message_parts, chat_history
import logging
from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json
from typing import List
from PIL import Image
import io
import base64
import numpy as np
import re


class NLPController(BaseController):
    
    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def resize_and_compress_image_b64(self, image_b64, max_size=(512, 512), quality=70):
        """Resize and compress a base64 image string to reduce token size."""
        logger = logging.getLogger("uvicorn.error")
        try:
            # Remove data URL prefix if present
            if image_b64.startswith("data:"):
                header, image_b64 = image_b64.split(",", 1)
            before_size = len(image_b64)
            logger.info(f"[ImageResize] Original image base64 size: {before_size / 1024:.2f} KB")
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            resized_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            after_size = len(resized_b64)
            logger.info(f"[ImageResize] Resized/compressed image base64 size: {after_size / 1024:.2f} KB")
            return f"data:image/jpeg;base64,{resized_b64}"
        except Exception as e:
            logger.error(f"[ImageResize] Failed to resize/compress image: {e}")
            return None

    def create_collection_name(self, project_id: str, user_id: str):
        return f"collection_{user_id}_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=str(project.project_id), user_id=str(project.user_id))
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=str(project.project_id), user_id=str(project.user_id))
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

        collection_name = self.create_collection_name(project_id=str(project.project_id), user_id=str(project.user_id))
        logger.info(f"Indexing into vector DB collection: {collection_name}")

        # --- Batch preparation ---
        texts = []
        metadata = []
        valid_ids = []
        doc_types = []

        for i, c in enumerate(chunks):
            # Treat table images as images
            chunk_type = c.chunk_metadata.get("type")
            is_table_image = c.chunk_metadata.get("is_table_image", False)
            if (chunk_type == "image" and c.chunk_text) or is_table_image:
                image_b64 = c.chunk_text
                image_b64_resized = self.resize_and_compress_image_b64(image_b64)
                if image_b64_resized is not None:
                    image_b64 = image_b64_resized
                else:
                    logger.warning(f"Skipping image chunk at index {i} due to resize failure.")
                    continue
                caption = c.chunk_metadata.get("caption")
                texts.append(image_b64)
                doc_types.append("image")
            else:
                texts.append(c.chunk_text)
                doc_types.append(chunk_type)
            metadata.append(c.chunk_metadata)
            valid_ids.append(chunks_ids[i])

        # --- Batch embedding with chunking ---
        MAX_BATCH = 96

        def batch_indices(indices, batch_size):
            for i in range(0, len(indices), batch_size):
                yield indices[i:i+batch_size]

        image_indices = [i for i, t in enumerate(doc_types) if t == "image" and texts[i]]
        text_indices = [i for i, t in enumerate(doc_types) if t != "image"]

        vectors = [None] * len(texts)

        
        for idx in image_indices:
            image_input = texts[idx]
            logger.info(f"Embedding single image at index {idx}")
            image_vector = self.embedding_client.embed_text(image_input, document_type="image")
            if image_vector is None:
                logger.error(f"Image embedding failed at index {idx}.")
            vectors[idx] = image_vector

        # Batch embed texts
        for batch in batch_indices(text_indices, MAX_BATCH):
            text_inputs = [texts[i] for i in batch]
            if text_inputs:
                logger.info(f"Embedding text batch of size {len(text_inputs)}")
                text_vectors = self.embedding_client.embed_text(text_inputs, document_type="document")
                if text_vectors is None:
                    logger.error("Text embedding failed.")
                    text_vectors = [None] * len(text_inputs)
                for idx, vec in zip(batch, text_vectors):
                    vectors[idx] = vec

        # Filter out any failed embeddings
        final_texts = []
        final_metadata = []
        final_vectors = []
        final_ids = []
        for i, vec in enumerate(vectors):
            if isinstance(vec, (list, tuple)) and len(vec) == self.embedding_client.embedding_size:
                final_texts.append(texts[i])
                final_metadata.append(metadata[i])
                final_vectors.append(vec)
                final_ids.append(valid_ids[i])
            else:
                logger.warning(f"Skipping chunk at index {i}: embedding is not a list or wrong size. Got type: {type(vec)}, value: {vec}")

        logger.info(f"Creating collection '{collection_name}' with embedding size {self.embedding_client.embedding_size}")
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        logger.info(f"Inserting {len(final_vectors)} vectors into collection '{collection_name}'")
        inserted = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=final_texts,
            metadata=final_metadata,
            vectors=final_vectors,
            record_ids=final_ids,
        )

        inserted_count = len(final_vectors) if inserted else 0
        logger.info(f"Inserted {inserted_count} vectors into collection '{collection_name}'.")

        return True

    def extract_page_number(self, query: str):
        # Simple regex to find "page 5" or "on page 12"
        match = re.search(r'page\s*(\d+)', query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 20, image_limit: int = 20):
        logger = logging.getLogger("uvicorn.error")
        logger.info(f"Searching vector DB for project {project.project_id} with query: {text}")

        collection_name = self.create_collection_name(project_id=str(project.project_id), user_id=str(project.user_id))
        vector = self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            logger.warning("No embedding vector generated for query.")
            return False

        # --- Text chunks ---
        text_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.type",
                    match=MatchValue(value="text")
                )
            ]
        )
        text_results = self.vectordb_client.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=text_filter
        )

        # --- Image chunks ---
        image_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.type",
                    match=MatchValue(value="image")
                )
            ]
        )
        image_results = self.vectordb_client.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=image_limit,
            query_filter=image_filter
        )

        # Convert to RetrievedDocument
        from models.db_schemes import RetrievedDocument
        def to_docs(results):
            return [
                RetrievedDocument(
                    score=result.score,
                    text=result.payload["text"],
                    metadata=result.payload.get("metadata", {})
                )
                for result in results
            ] if results else []

        text_docs = to_docs(text_results)
        image_docs = to_docs(image_results)

        # --- Rerank images using captions ---
        image_docs = self.rerank_images_with_captions(text, image_docs)

        # --- Boost image scores if query is image-related ---
        if self.is_image_query(text):
            boost_factor = 1.2 
            for doc in image_docs:
                doc.score *= boost_factor

        # --- Boost by page number if specified ---
        page_number = self.extract_page_number(text)
        if page_number is not None:
            boost_factor = 1.5  # You can tune this
            for doc in text_docs + image_docs:
                doc_page = doc.metadata.get("page_number")
                if doc_page is not None and str(doc_page) == str(page_number):
                    doc.score *= boost_factor

        # --- Combine and rerank all docs with FlagEmbedding ---
        combined = text_docs + image_docs
        if combined:
            combined = self.rerank_with_flagembedding(text, combined)
            combined = combined[:20]  # Rerank top 20, then send top 10 to LLM

        return combined

    def answer_rag_question(self, project: Project, query: str, limit: int = 10, chat_history=None):
        logger = logging.getLogger("uvicorn.error")
        logger.info(f"Starting RAG answer for project {project.project_id} with query: {query}")
        answer, full_prompt, chat_history_out = None, None, None

        # step1: retrieve related documents (now always gets up to 10 after rerank)
        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=20,  # fetch more for reranking
            image_limit=10
        )
        if not retrieved_documents or len(retrieved_documents) == 0:
            logger.warning("No documents retrieved from vector DB.")
            return answer, full_prompt, chat_history_out

        logger.info(f"Retrieved {len(retrieved_documents)} documents from vector DB.")

        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")
        message_parts = []

        for idx, doc in enumerate(retrieved_documents[:10]):  # Only send top 10 to LLM
            doc_type = doc.metadata.get("type", "text")
            is_table_image = doc.metadata.get("is_table_image", False)
            page_number = doc.metadata.get("page_number")
            if doc_type == "image" or is_table_image:
                logger.info(f"Adding image document {idx} to prompt.")
                image_b64 = doc.text
                caption = doc.metadata.get("caption")
                if not image_b64.startswith("data:"):
                    image_b64 = f"data:image/png;base64,{image_b64}"
                message_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_b64}
                })
                # Add page number and caption/context
                page_info = f"(Page: {page_number if page_number is not None else 'N/A'})"
                if caption:
                    message_parts.append({
                        "type": "text",
                        "text": f"Image {idx+1} {page_info}: {caption}"
                    })
                else:
                    message_parts.append({
                        "type": "text",
                        "text": f"Image {idx+1} {page_info}"
                    })
            else:
                logger.info(f"Adding text document {idx} to prompt.")
                # Add prev_context and next_context if available
                prev_context = doc.metadata.get("prev_context")
                next_context = doc.metadata.get("next_context")
                chunk_text = doc.text
                context_parts = []
                if prev_context:
                    context_parts.append(f"[Previous context]\n{prev_context}")
                context_parts.append(f"[Main chunk]\n{chunk_text}")
                if next_context:
                    context_parts.append(f"[Next context]\n{next_context}")
                full_chunk_text = "\n\n".join(context_parts)
                message_parts.append({
                    "type": "text",
                    "text": self.template_parser.get("rag", "document_prompt", {
                        "doc_num": idx + 1,
                        "chunk_text": full_chunk_text,
                        "page_number": page_number if page_number is not None else "N/A"
                    })
                })

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})
        message_parts.append({"type": "text", "text": footer_prompt})
        logger.info("Prompt for LLM constructed.")

        # step3: Construct Generation Client Prompts
        chat_history_out = [
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
            chat_history=chat_history_out
        )

        if answer:
            logger.info("Received answer from generation client.")
        else:
            logger.warning("No answer received from generation client.")

        return answer, message_parts, chat_history_out

    def is_image_query(self, query: str) -> bool:
        image_keywords = [
            "image", "figure", "diagram", "chart", "picture", "photo", "see", "show",
            "illustration", "graph", "table", "visual", "plot", "map", "screenshot",
            "scan", "drawing", "sketch", "look like"
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in image_keywords)

    def rerank_images_with_captions(self, query, image_docs):
        """
        Rerank image_docs by combining their original vector score and the semantic similarity
        between the query and the image's caption (if available).
        """
        # Embed the query
        query_vec = self.embedding_client.embed_text(query, document_type="query")
        reranked = []
        for doc in image_docs:
            caption = doc.metadata.get("caption", "")
            if caption:
                caption_vec = self.embedding_client.embed_text(caption, document_type="document")
                # Compute cosine similarity between query and caption
                def cosine_similarity(a, b):
                    a = np.array(a)
                    b = np.array(b)
                    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                sim = cosine_similarity(query_vec, caption_vec)
                # Combine with image score (tune weights as needed)
                combined_score = 0.7 * doc.score + 0.3 * sim
            else:
                combined_score = doc.score
            reranked.append((doc, combined_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked]

    def rerank_with_flagembedding(self, query: str, docs: list):
        from FlagEmbedding import FlagReranker

        # You may want to cache the reranker instance in production
        reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)

        pairs = [(query, doc.text) for doc in docs]
        scores = reranker.compute_score(pairs)

        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked]
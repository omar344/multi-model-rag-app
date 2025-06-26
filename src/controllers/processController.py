from .BaseController import BaseController
from .ProjectController import ProjectController
import os
import logging
import base64
from typing import List
from models import ProcessingEnum
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from unstructured.partition.text import partition_text
from unstructured.documents.elements import (
    CompositeElement, Table, Image, NarrativeText
)
from langchain.schema import Document

logger = logging.getLogger("uvicorn.error")

class ProcessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1].lower()

    def get_file_content(self, file_id: str) -> List[Document]:
        file_path = os.path.join(self.project_path, file_id)
        ext = self.get_file_extension(file_id)

        try:
            if ext == ProcessingEnum.PDF.value:
                return self.process_pdf(file_path)
            elif ext in [ProcessingEnum.JPG.value, ProcessingEnum.JPEG.value, ProcessingEnum.PNG.value]:
                return self.process_image(file_path)
            elif ext == ProcessingEnum.TXT.value:
                return self.process_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error processing file {file_id}: {e}")
            raise ValueError(f"Processing failed: {e}")

    def extract_essential_metadata(self, metadata):
        return {
            k: getattr(metadata, k, None)
            for k in ["page_number", "filetype", "filename", "languages"]
            if hasattr(metadata, k)
        }

    def classify_element(self, el, elements=None, idx=None):
        meta = self.extract_essential_metadata(el.metadata)

        if isinstance(el, Table):
            # If the table has a screenshot (image_base64), treat as image
            image_b64 = getattr(el.metadata, "image_base64", None)
            if image_b64:
                # Try to find a caption or nearby text
                caption = None
                if elements is not None and idx is not None:
                    for offset in [-1, 1]:
                        neighbor_idx = idx + offset
                        if 0 <= neighbor_idx < len(elements):
                            neighbor = elements[neighbor_idx]
                            if isinstance(neighbor, NarrativeText):
                                caption = neighbor.text
                                break
                return Document(
                    page_content=image_b64,
                    metadata={**meta, "type": "image", "caption": caption, "is_table_image": True},
                )
            else:
                # Fallback: treat as text if no image
                return Document(
                    page_content=str(el),
                    metadata={**meta, "type": "table"},
                )
        elif isinstance(el, NarrativeText):
            return Document(
                page_content=el.text,
                metadata={**meta, "type": "text"},
            )
        elif isinstance(el, Image):
            # Try to find a caption or nearby text
            caption = None
            if elements is not None and idx is not None:
                for offset in [-1, 1]:
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(elements):
                        neighbor = elements[neighbor_idx]
                        if isinstance(neighbor, NarrativeText):
                            caption = neighbor.text
                            break
            image_b64 = getattr(el.metadata, "image_base64", None)
            if image_b64 is None:
                image_b64 = ""  # Ensure always a string
            return Document(
                page_content=image_b64,
                metadata={**meta, "type": "image", "caption": caption, "is_table_image": False},
            )
        return None

    def flatten_elements(self, elements):
        docs = []
        seen_ids = set()
        max_paragraphs_per_chunk = 4  # Larger chunk size
        overlap_paragraphs = 2        # More overlap
        text_chunks = []
        text_metadatas = []
        # First, flatten all NarrativeText into paragraph chunks with context
        for idx, el in enumerate(elements):
            # Use id(el) to uniquely identify each element object
            if isinstance(el, CompositeElement) and hasattr(el.metadata, "orig_elements"):
                orig_elements = getattr(el.metadata, "orig_elements", None)
                if orig_elements is not None:
                    for sub in orig_elements:
                        if id(sub) not in seen_ids:
                            if isinstance(sub, NarrativeText):
                                paragraphs = [p.strip() for p in sub.text.split('\n\n') if p.strip()]
                                i = 0
                                while i < len(paragraphs):
                                    chunk = "\n\n".join(paragraphs[i:i+max_paragraphs_per_chunk])
                                    meta = self.extract_essential_metadata(sub.metadata)
                                    meta.update({"type": "text"})
                                    text_chunks.append(chunk)
                                    text_metadatas.append(meta)
                                    i += max_paragraphs_per_chunk - overlap_paragraphs
                            else:
                                doc = self.classify_element(sub, elements=elements, idx=idx)
                                if doc:
                                    docs.append(doc)
                            seen_ids.add(id(sub))
            else:
                if id(el) not in seen_ids:
                    if isinstance(el, NarrativeText):
                        paragraphs = [p.strip() for p in el.text.split('\n\n') if p.strip()]
                        i = 0
                        while i < len(paragraphs):
                            chunk = "\n\n".join(paragraphs[i:i+max_paragraphs_per_chunk])
                            meta = self.extract_essential_metadata(el.metadata)
                            meta.update({"type": "text"})
                            text_chunks.append(chunk)
                            text_metadatas.append(meta)
                            i += max_paragraphs_per_chunk - overlap_paragraphs
                    else:
                        doc = self.classify_element(el, elements=elements, idx=idx)
                        if doc:
                            docs.append(doc)
                    seen_ids.add(id(el))
        # Add context window to each text chunk
        for i, (chunk, meta) in enumerate(zip(text_chunks, text_metadatas)):
            prev_context = text_chunks[i-1] if i > 0 else None
            next_context = text_chunks[i+1] if i < len(text_chunks)-1 else None
            meta = meta.copy()
            if prev_context:
                meta["prev_context"] = prev_context
            if next_context:
                meta["next_context"] = next_context
            # Ensure chunk is always a string
            docs.append(Document(page_content=chunk if chunk is not None else "", metadata=meta))
        return docs

    def process_pdf(self, path: str) -> List[Document]:
        elements = partition_pdf(
            filename=path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            include_metadata=True,
            chunking_strategy="by_title",
            max_characters=1000,
            overlap=100,
        )
        return self.flatten_elements(elements)

    def process_image(self, path: str) -> List[Document]:
        elements = partition_image(filename=path, include_metadata=True)
        return self.flatten_elements(elements)

    def process_text(self, path: str) -> List[Document]:
        elements = partition_text(filename=path, include_metadata=True)
        return self.flatten_elements(elements)

    def process_file_content(self, file_content: List[Document], file_id: str) -> List[Document]:
        if not file_content:
            logger.warning(f"No content extracted from file: {file_id}")
            return []

        for doc in file_content:
            # Base64 encode any raw image bytes (if any slipped through)
            for k, v in list(doc.metadata.items()):
                if isinstance(v, bytes):
                    doc.metadata[k] = base64.b64encode(v).decode("utf-8")

        logger.info(
            f"Processed {len(file_content)} elements from file {file_id} "
            f"({sum(d.metadata.get('type') == 'text' for d in file_content)} text, "
            f"{sum(d.metadata.get('type') == 'table' for d in file_content)} tables, "
            f"{sum(d.metadata.get('type') == 'image' for d in file_content)} images)"
        )

        return file_content

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

    def classify_element(self, el):
        meta = self.extract_essential_metadata(el.metadata)

        if isinstance(el, Table):
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
            return Document(
                page_content=getattr(el.metadata, "image_base64", None),
                metadata={
                    **meta,
                    "type": "image"
                },
            )
        return None

    def flatten_elements(self, elements):
        docs = []
        seen_ids = set()
        for el in elements:
            # Use id(el) to uniquely identify each element object
            if isinstance(el, CompositeElement) and hasattr(el.metadata, "orig_elements"):
                for sub in el.metadata.orig_elements:
                    if id(sub) not in seen_ids:
                        doc = self.classify_element(sub)
                        if doc:
                            docs.append(doc)
                        seen_ids.add(id(sub))
            else:
                if id(el) not in seen_ids:
                    doc = self.classify_element(el)
                    if doc:
                        docs.append(doc)
                    seen_ids.add(id(el))
        return docs

    def process_pdf(self, path: str) -> List[Document]:
        elements = partition_pdf(
            filename=path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            include_metadata=True,
            chunking_strategy="by_title",
            max_characters=400,
            overlap=20,
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

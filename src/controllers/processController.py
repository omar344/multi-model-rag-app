from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from models import ProcessingEnum
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredImageLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import base64
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.partition.image import partition_image

logger = logging.getLogger("uvicorn.error")

class ProcessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        
    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1].lower()
    
    def get_file_loader(self, file_id: str):
        file_ext = self.get_file_extension(file_id)
        file_path = os.path.join(self.project_path, file_id)
        
        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        
        if file_ext in [ProcessingEnum.JPG.value, ProcessingEnum.JPEG.value, ProcessingEnum.PNG.value]:
            return UnstructuredImageLoader(file_path)
        
        if file_ext == ProcessingEnum.PDF.value:
            return UnstructuredPDFLoader(file_path)
        
        return None
    
    def get_file_content(self, file_id: str):
        file_ext = self.get_file_extension(file_id)
        file_path = os.path.join(self.project_path, file_id)
        
        try:
            if file_ext == ProcessingEnum.PDF.value:
                # Use partition_pdf for PDFs to extract rich features
                return self.process_pdf_with_partition(file_path)
            else:
                # Use standard loaders for other file types
                file_loader = self.get_file_loader(file_id=file_id)
                if not file_loader:
                    logger.error(f"No loader found for file: {file_id}")
                    raise ValueError(f"Unsupported file type for file: {file_id}")
                
                file_content = file_loader.load()
                
                # Add type information to non-PDF files
                for doc in file_content:
                    if file_ext == ProcessingEnum.TXT.value:
                        doc.metadata["type"] = "text"
                    elif file_ext in [ProcessingEnum.JPG.value, ProcessingEnum.JPEG.value, ProcessingEnum.PNG.value]:
                        doc.metadata["type"] = "image"
                
                return file_content
        except Exception as e:
            logger.error(f"Error loading file {file_id}: {e}")
            raise ValueError(f"Failed to load file {file_id}.")
    
    def process_pdf_with_partition(self, file_path):
        """Process PDF using unstructured's partition_pdf for rich feature extraction"""
        try:
            # Extract elements with partition_pdf - always use hi_res strategy
            elements = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",  # Always use high resolution processing
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=4000,
                combine_text_under_n_chars=1000,
                new_after_n_chars=3000,
            )
            
            # Process different element types
            documents = []
            tables = []
            texts = []
            images = []
            
            # First pass: categorize elements and extract base images
            for element in elements:
                element_type = self.determine_element_type(element)
                
                if element_type == "table":
                    tables.append(element)
                elif element_type == "image":
                    images.append(element)
                elif "CompositeElement" in str(type(element)):
                    texts.append(element)
                    # Extract images from composite elements
                    if hasattr(element, "metadata") and hasattr(element.metadata, "orig_elements"):
                        for sub_el in element.metadata.orig_elements:
                            if "Image" in str(type(sub_el)):
                                images.append(sub_el)
                else:
                    texts.append(element)
            
            # Second pass: convert to LangChain documents
            # Process text and composite elements
            for element in texts:
                metadata = {
                    "source": file_path,
                    "type": "text"
                }
                
                if hasattr(element, "metadata"):
                    if hasattr(element.metadata, "page_number"):
                        metadata["page"] = element.metadata.page_number
                    if "CompositeElement" in str(type(element)):
                        metadata["type"] = "composite"
                
                documents.append(
                    Document(
                        page_content=str(element),
                        metadata=metadata
                    )
                )
            
            # Process tables
            for element in tables:
                metadata = {
                    "source": file_path,
                    "type": "table"
                }
                
                if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
                    metadata["page"] = element.metadata.page_number
                
                documents.append(
                    Document(
                        page_content=str(element),
                        metadata=metadata
                    )
                )
            
            # Process images and extract base64 content
            for element in images:
                metadata = {
                    "source": file_path,
                    "type": "image"
                }
                
                if hasattr(element, "metadata"):
                    if hasattr(element.metadata, "page_number"):
                        metadata["page"] = element.metadata.page_number
                    if hasattr(element.metadata, "image_base64"):
                        metadata["image_data"] = element.metadata.image_base64
                
                image_content = "Image from document"
                documents.append(
                    Document(
                        page_content=image_content,
                        metadata=metadata
                    )
                )
            
            # Get additional images from composite elements separately
            additional_images = self.get_images_base64(elements)
            for i, image_data in enumerate(additional_images):
                if image_data not in [d.metadata.get("image_data") for d in documents if d.metadata.get("type") == "image"]:
                    documents.append(
                        Document(
                            page_content=f"Additional image {i+1} from document",
                            metadata={
                                "source": file_path,
                                "type": "image",
                                "image_data": image_data
                            }
                        )
                    )
            
            logger.info(f"Extracted {len(texts)} text elements, {len(tables)} tables, and {len(images)} images from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error in PDF processing: {e}")
            raise ValueError(f"Failed to process PDF file: {e}")
    
    def determine_element_type(self, element):
        """Determine the element type based on its class name"""
        element_class = str(type(element))
        
        if "Table" in element_class:
            return "table"
        elif "Image" in element_class:
            return "image"
        elif "Title" in element_class:
            return "heading"
        elif "CompositeElement" in element_class:
            return "composite"
        else:
            return "text"
    
    def get_images_base64(self, chunks):
        """Extract base64 image data from composite elements"""
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)):
                            if hasattr(el, "metadata") and hasattr(el.metadata, "image_base64"):
                                images_b64.append(el.metadata.image_base64)
        return images_b64
    
    def process_file_content(self, file_content: list,
                             file_id: str, chunk_size: int=100, overlap_size: int=20):
        """Process file content into chunks based on content type"""
        if not file_content:
            logger.warning(f"No content extracted from file: {file_id}")
            return []
        
        # Separate content by type
        text_content = []
        non_text_content = []
        
        for doc in file_content:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "text" or doc_type == "composite":
                text_content.append(doc)
            else:
                non_text_content.append(doc)
        
        # Apply text splitting only to text documents
        if text_content:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
                length_function=len
            )
            
            text_chunks = splitter.split_documents(text_content)
        else:
            text_chunks = []
        
        # Combine all chunks
        all_chunks = text_chunks + non_text_content
        
        # Process binary data for JSON serialization
        for chunk in all_chunks:
            for key, value in list(chunk.metadata.items()):
                if isinstance(value, bytes):
                    chunk.metadata[key] = base64.b64encode(value).decode('utf-8')
        
        logger.info(f"Processed {len(all_chunks)} elements from file: {file_id}")
        return all_chunks
from enum import Enum

class ResponseSignal(Enum):
    
    FILE_TYPE_NOT_SUPPORTED = "file type not supported"
    FILE_SIZE_TOO_LARGE = "file size too large"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "failed"
    PROCESSING_SUCCESS = "processing success"
    PROCESSING_FAILED = "processing failed"
    PROJECT_NOT_FOUND_ERROR = "project not found"
    VECTORDB_INSERTION_ERROR = "vectordb insertion error"
    VECTORDB_INSERTION_SUCCESS = "vectordb insertion success"
    VECTORDB_COLLECTION_RETRIEVED = "retrieved collection successfully"
    VECTORDB_SEARCH_SUCCESS = "searched for collection successfully"
    VECTORDB_SEARCH_ERROR = " failed to search for collection"
    RAG_ANSWER_ERROR = "failed to generate an answer"
    RAG_ANSWER_SUCCESS = "answer generated successfully"
    
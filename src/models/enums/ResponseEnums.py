from enum import Enum

class ResponseSignal(Enum):
    
    FILE_TYPE_NOT_SUPPORTED = "file type not supported"
    FILE_SIZE_TOO_LARGE = "file size too large"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "failed"
    
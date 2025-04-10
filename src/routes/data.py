from fastapi import APIRouter, FastAPI, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
from  models import ResponseSignal
import logging
from .schemes.dataScheme import processRequest

logger = logging.getLogger("uvicorn.error")
data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1/data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile,
                      app_settings:Settings = Depends(get_settings)):

#validate the file properties
    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_upload_file(file=file)
    if not is_valid:  
        return JSONResponse(
            status_code = status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
                }
            )
    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name = file.filename,
        project_id = project_id
    )

    try:

        async with aiofiles.open(file_path, 'wb') as buffer:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNCK_SIZE):
                await buffer.write(chunk)
    
    except Exception as e:  
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )
   

    return JSONResponse(
    content={
        "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
        "file id": file_id
        }
    )

@data_router.post("/process/{project_id}")
async def process_endpoint(project_id: str, process_request: processRequest):
    
    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    
    process_controller = ProcessController(project_id=project_id)
    
    try:
        file_content = process_controller.get_file_content(file_id=file_id)
        
        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )
        
        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.PROCESSING_FAILED.value
                }
            )
        
        # Serialize chunks into JSON-compatible format with type information
        serialized_chunks = [
            {
                "content": chunk.page_content,
                "type": chunk.metadata.get("type", "text"),  # Include document type
                "metadata": {
                    k: v for k, v in chunk.metadata.items() 
                    if k != "image_data"  # Exclude image_data from general metadata
                }
            }
            for chunk in file_chunks
        ]
        
        # Separately handle image data to make it accessible but not duplicate it
        for i, chunk in enumerate(file_chunks):
            if chunk.metadata.get("type") == "image" and "image_data" in chunk.metadata:
                serialized_chunks[i]["image_data"] = chunk.metadata["image_data"]
        
        # Return serialized chunks as JSON response
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseSignal.PROCESSING_SUCCESS.value,
                "chunks": serialized_chunks
            }
        )
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROCESSING_FAILED.value,
                "error": str(e)
            }
        )
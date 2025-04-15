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
        "file_id": file_id
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
            file_id=file_id
        )

        if not file_chunks:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.PROCESSING_FAILED.value}
            )

        # Serialize into JSON format
        serialized_chunks = []
        for chunk in file_chunks:
            serialized = {
                "content": chunk.page_content,
                "metadata": {
                    k: v for k, v in chunk.metadata.items() if k != "image_base64"
                }
            }

            # Only attach image base64 separately if it's an image
            if chunk.metadata.get("type") == "image" and "image_base64" in chunk.metadata:
                serialized["image_base64"] = chunk.metadata["image_base64"]

            serialized_chunks.append(serialized)

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
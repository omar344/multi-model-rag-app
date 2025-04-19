from fastapi import APIRouter, FastAPI, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController
from controllers import ProjectController
import aiofiles
from  models import ResponseSignal
import logging
from models.ProjectModel import ProjectModel


logger = logging.getLogger("uvicorn.error")
data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1/data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: str, file: UploadFile,
                      app_settings:Settings = Depends(get_settings)):


    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project =await  project_model.get_project_or_great_one(
        project_id = project_id
    )
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
        "file_id": file_id,
        "project_id": str(project._id)

        }
    )


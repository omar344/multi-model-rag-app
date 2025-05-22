from fastapi import APIRouter, UploadFile, Request, status, Depends
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController, NLPController
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.ChunkModel import ChunkModel
from models.db_schemes import Asset, DataChunk
from models.enums.AssetTypeEnum import AssetTypeEnum
from models.enums.ProcessingEnum import ProcessingEnum
from models import ResponseSignal
from routes.schemes.nlpScheme import SearchRequest
from routes.auth import get_current_user

import os
import aiofiles
import logging
import uuid

logger = logging.getLogger("uvicorn.error")

rag_router = APIRouter(
    prefix="/api/v1/rag",
    tags=["api_v1", "rag"]
)

@rag_router.post("/upload_file")
async def upload_and_index(
    request: Request,
    file: UploadFile,
    app_settings: Settings = Depends(get_settings),
    current_user=Depends(get_current_user)
):
    logger.info("Starting file upload and indexing process.")

    # --- Auto-generate project_id
    project_id = uuid.uuid4().hex
    logger.info(f"Generated project_id: {project_id}")

    # --- Project setup
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    logger.info("ProjectModel instance created.")
    project = await project_model.get_project_or_create_one(
        project_id=project_id,
        user_id=current_user.id
    )
    logger.info(f"Project retrieved or created: {project.id}")

    # --- Validate and save file
    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)
    logger.info(f"File validation result: {is_valid}, signal: {result_signal}")
    if not is_valid:
        logger.error("File validation failed.")
        return JSONResponse(status_code=400, content={"signal": result_signal})

    file_path, file_id = data_controller.generate_unique_filepath(file.filename, project_id)
    logger.info(f"Generated file path: {file_path}, file_id: {file_id}")
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
        logger.info("File saved successfully.")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(status_code=400, content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value})

    # --- Store in DB
    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    logger.info("AssetModel instance created.")
    asset_record = await asset_model.create_asset(Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path),
        user_id=current_user.id
    ), user_id=current_user.id)
    logger.info(f"Asset record created: {asset_record.id}")

    # --- Process file into chunks
    process_controller = ProcessController(project_id=project_id)
    logger.info("ProcessController instance created.")
    content = process_controller.get_file_content(file_id=file_id)
    logger.info("File content loaded.")
    chunks = process_controller.process_file_content(file_content=content, file_id=file_id)
    logger.info(f"File processed into {len(chunks) if chunks else 0} chunks.")

    if not chunks:
        logger.error("No chunks generated from file.")
        return JSONResponse(status_code=400, content={"signal": ResponseSignal.PROCESSING_FAILED.value})

    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
    logger.info("ChunkModel instance created.")
    ext = process_controller.get_file_extension(file_id)
    file_type_enum = ProcessingEnum(ext)

    chunk_records = [
        DataChunk(
            chunk_text=chunk.page_content,
            chunk_metadata=chunk.metadata,
            chunk_order=i+1,
            chunk_project_id=project.id,
            chunk_asset_id=asset_record.id,
            file_type=file_type_enum.value,
            user_id=current_user.id
        )
        for i, chunk in enumerate(chunks)
    ]
    logger.info(f"Prepared {len(chunk_records)} chunk records for insertion.")

    await chunk_model.insert_many_chunks(chunks=chunk_records, user_id=current_user.id)  # <-- ENFORCE OWNERSHIP
    logger.info("Chunks inserted into database.")

    # --- Index to vector DB
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )
    logger.info("NLPController instance created.")

    is_indexed = nlp_controller.index_into_vector_db(
        project=project,
        chunks=chunk_records,
        do_reset=True,
        chunks_ids=list(range(len(chunk_records)))
    )
    logger.info(f"Indexing to vector DB result: {is_indexed}")

    if not is_indexed:
        logger.error("Vector DB insertion failed.")
        return JSONResponse(status_code=400, content={"signal": ResponseSignal.VECTORDB_INSERTION_ERROR.value})

    logger.info("File upload and indexing process completed successfully.")
    return JSONResponse(content={
        "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
        "project_id": project_id,
        "file_id": asset_record.asset_name,
        "inserted_chunks": len(chunk_records)
    })


@rag_router.post("/ask/{project_id}")
async def ask_question(request: Request, project_id: str, search_request: SearchRequest, current_user=Depends(get_current_user)):
    logger.info(f"Received question for project_id: {project_id}")

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    logger.info("ProjectModel instance created.")
    project = await project_model.get_project_or_create_one(
        project_id=project_id,
        user_id=current_user.id
    )
    logger.info(f"Project retrieved or created: {project.id}")

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )
    logger.info("NLPController instance created.")

    answer, full_prompt, chat_history = nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )
    logger.info("RAG question answered.")

    if not answer:
        logger.error("No answer generated by RAG pipeline.")
        return JSONResponse(status_code=400, content={"signal": ResponseSignal.RAG_ANSWER_ERROR.value})

    logger.info("Returning answer to client.")
    return JSONResponse(content={
        "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
        "answer": answer,
        "full_prompt": full_prompt,
        "chat_history": chat_history
    })

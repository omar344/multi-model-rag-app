from .BaseDataModel import BaseDataModel
from models.db_schemes import DataChunk
from models.enums.DataBaseEnum import DataBaseEnum
from bson import ObjectId
from pymongo import InsertOne
from pymongo.errors import PyMongoError
import logging

class ChunkModel(BaseDataModel):
    
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHUNK_NAME.value]
        
    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_CHUNK_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_CHUNK_NAME.value]
            indexes = DataChunk.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"]
                )

    async def create_chunk(self, chunk: DataChunk):
        try:
            result = await self.collection.insert_one(chunk.dict())
            chunk._id = result.inserted_id
            return chunk
        except PyMongoError as e:
            logging.error(f"Error inserting chunk: {e}")
            raise

    async def get_chunk(self, chunk_id: str):
        try:
            result = await self.collection.find_one({
                "_id": ObjectId(chunk_id)
            })
            if result is None:
                return None
            return DataChunk(**result)
        except PyMongoError as e:
            logging.error(f"Error retrieving chunk with ID {chunk_id}: {e}")
            raise

    async def insert_many_chunks(self, chunks: list, batch_size: int = 100):
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                operations = [
                    InsertOne(chunk.dict(by_alias=True, exclude_unset=True))
                    for chunk in batch
                ]
                await self.collection.bulk_write(operations)
            return len(chunks)
        except PyMongoError as e:
            logging.error(f"Error inserting chunks: {e}")
            raise

    async def delete_chunks_by_project_id(self, project_id: ObjectId):
        try:
            result = await self.collection.delete_many({
                "chunk_project_id": project_id
            })
            logging.info(f"Deleted {result.deleted_count} chunks for project ID {project_id}")
            return result.deleted_count
        except PyMongoError as e:
            logging.error(f"Error deleting chunks for project ID {project_id}: {e}")
            raise

    async def get_project_chunks(self, project_id: ObjectId, page_no: int = 1, page_size: int = 50):
        try:
            records = await self.collection.find({
                "chunk_project_id": project_id
            }).skip(
                (page_no - 1) * page_size
            ).limit(page_size).to_list(length=None)

            return [DataChunk(**record) for record in records]
        except PyMongoError as e:
            logging.error(f"Error retrieving chunks for project ID {project_id}: {e}")
            raise

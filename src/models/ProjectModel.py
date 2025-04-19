from models.BaseDataModel import BaseDataModel
from helpers.config import get_settings
from .enums.DatabaseEnum import DatabaseEnum
from models.db_schemas import Project



class ProjectModel(BaseDataModel):

    def __init__(self):
        super().__init__(db_client = get_settings().MONGODB_DATABASE)
        self.collection = self.db_client[DatabaseEnum.PROJECT.value]
    
    def create_project(self, project: Project):
        result = self.collection.insert_one(project.dict())
        project._id = result.inserted_id
        return project
    
    async def get_project(self, project_id: str):
        record = await self.collection.find_one(
            {"project_id": project_id}
        )
        if record is None:
            project = Project(project_id=project_id)
            project = self.create_project(project)
            return project
        return Project(**record) #Create a new Project instance with the record data
    
    async def get_all_project(self, page: int=1, page_size: int=10):

        total_documents = await self.collection.count_documents({})
       
        total_pages = total_documents // page_size
        if total_documents % page_size > 0:
            total_pages += 1

        cursor = self.collection.find().skip((page-1) * page_size).limit(page_size)
        projects = []
        async for document in cursor:
            projects.append(
                Project(**document)
            )        
        
        return projects, total_pages
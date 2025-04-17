from BaseDataModdel import BaseDataModel
from helpers.config import get_settings
from .enums.DatabaseEnum import DatabaseEnum



class ProjectModel(BaseDataModel):

    def __init__(self):
        super().__init__(db_client = get_settings().MONGODB_DATABASE)
        self.collection_name = self.db_client[DatabaseEnum.PROJECT.value]
        
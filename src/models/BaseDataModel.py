from helpers.config import get_settings, Settings
class BaseDataModel:
    def __init__(self, db_client):
        self.db_client = db_client
        self.app_name = get_settings().APP_NAME


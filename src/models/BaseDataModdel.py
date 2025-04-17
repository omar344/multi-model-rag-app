from helpers.config import get_settings, Settings
class BaseDataModel:

    def __init__(self):
        self.db_client = db_client = get_settings().MONGODB_DATABASE
        self.app = app = get_settings().APP_NAME
        
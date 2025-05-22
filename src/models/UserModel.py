from .BaseDataModel import BaseDataModel
from .db_schemes.user import User
from pymongo.errors import DuplicateKeyError

class UserModel(BaseDataModel):
    def __init__(self, db_client):
        super().__init__(db_client=db_client)
        self.collection = self.db_client["users"]

    @classmethod
    async def create_instance(cls, db_client):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        await self.collection.create_index("username", unique=True)
        await self.collection.create_index("email", unique=True)

    async def create_user(self, user: User):
        result = await self.collection.insert_one(user.dict(by_alias=True, exclude_unset=True))
        user.id = result.inserted_id
        return user

    async def get_user_by_username(self, username: str):
        record = await self.collection.find_one({"username": username})
        return User(**record) if record else None

    async def get_user_by_email(self, email: str):
        record = await self.collection.find_one({"email": email})
        return User(**record) if record else None
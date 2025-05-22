from fastapi import APIRouter, HTTPException, status, Request, Depends, Header
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from models.UserModel import UserModel
from models.db_schemes.user import User
from routes.schemes.authScheme import RegisterRequest, LoginRequest
from helpers.config import get_settings

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

settings = get_settings()
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@auth_router.post("/register")
async def register_user(request: Request, req: RegisterRequest):
    db = request.app.db_client
    user_model = await UserModel.create_instance(db)
    if await user_model.get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    if await user_model.get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    hashed_pw = pwd_context.hash(req.password)
    user = User(username=req.username, email=req.email, hashed_password=hashed_pw)
    await user_model.create_user(user)
    return {"msg": "User registered"}

@auth_router.post("/login")
async def login(request: Request, req: LoginRequest):
    db = request.app.db_client
    user_model = await UserModel.create_instance(db)
    # Check if input is email or username
    if "@" in req.identifier:
        user = await user_model.get_user_by_email(req.identifier)
    else:
        user = await user_model.get_user_by_username(req.identifier)
    if not user or not pwd_context.verify(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username, "user_id": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}

async def get_current_user(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        db = request.app.db_client
        user_model = await UserModel.create_instance(db)
        user = await user_model.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@auth_router.get("/me")
async def get_me(current_user=Depends(get_current_user)):
    # Return only safe user info
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "is_active": current_user.is_active,
        "is_admin": current_user.is_admin,
    }
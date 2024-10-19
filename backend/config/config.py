# config/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    MONGO_URI = os.getenv('MONGO_URI')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
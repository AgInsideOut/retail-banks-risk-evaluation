import os

class Config:
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.joblib')
    DEBUG = os.getenv('DEBUG', False)
    PORT = int(os.getenv('PORT', 8080))
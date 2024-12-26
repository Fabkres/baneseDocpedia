from pymongo import MongoClient

# Configurar conexão com o MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Ajuste conforme necessário
db = client["BDDocpedIA"]
feedback_collection = db["feedback"]

import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Get env vars
uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DATABASE")

print("Loaded DB name:", db_name)  # debug

# Connect
client = MongoClient(uri)
db = client[db_name]

print("Connected to DB:", db_name)
print("Collections:", db.list_collection_names())

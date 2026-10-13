import joblib
import io
from datetime import datetime
from pymongo import MongoClient
import gridfs

class ModelGridFSHandler:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="nhs_conditions_db"):
        """Initialize MongoDB connection, database, GridFS, and models collection."""
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        self.models_collection = self.db["models"]

    def save_model(self, model_object, vectorizer_object, model_name, model_type, labels, metrics):
        """Serialize, compress, and save the model and vectorizer into GridFS, then store metadata."""
        
        # 1. Serialize model and vectorizer together into an in-memory buffer
        buffer = io.BytesIO()
        joblib.dump({"model": model_object, "vectorizer": vectorizer_object}, buffer)
        buffer.seek(0)

        # 2. Upload the binary file to MongoDB GridFS
        file_id = self.fs.put(buffer.getvalue(), filename=f"{model_name}.joblib")

        # 3. Prepare the metadata document matching the project schema requirements[cite: 1]
        model_doc = {
            "name": model_name,
            "type": model_type,
            "gridfs_id": file_id,
            "labels": labels,
            "metrics": metrics,
            "created": datetime.utcnow()
        }

        # 4. Insert the metadata document into the 'models' collection[cite: 1]
        res = self.models_collection.insert_one(model_doc)
        print(f" Model '{model_name}' successfully saved to GridFS with ID: {file_id}")
        return res.inserted_id

    def load_latest_model(self, model_name):
        """Load the latest trained model and vectorizer from GridFS using the model name."""
        
        # Find the latest model document sorted by creation date descending
        doc = self.models_collection.find_one({"name": model_name}, sort=[("created", -1)])
        if not doc:
            raise FileNotFoundError(f"No model found with name: {model_name}")

        # Retrieve file data from GridFS using the stored gridfs_id reference[cite: 1]
        file_data = self.fs.get(doc["gridfs_id"]).read()
        buffer = io.BytesIO(file_data)
        saved_artifacts = joblib.load(buffer)
        
        return saved_artifacts["model"], saved_artifacts["vectorizer"], doc["labels"]
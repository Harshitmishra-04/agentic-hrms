import os
import json
import joblib

_pipeline = None
_metadata = None

def load_pipeline():
    global _pipeline
    if _pipeline is None:
        model_path = os.path.join("models", "v1", "attrition_pipeline.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model pipeline not found at {model_path}")
        _pipeline = joblib.load(model_path)
    return _pipeline

def load_metadata():
    global _metadata
    if _metadata is None:
        metadata_path = os.path.join("models", "v1", "metadata.json")
        if not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, "r") as f:
            _metadata = json.load(f)
    return _metadata

from transformers import pipeline
from sentence_transformers import SentenceTransformer


def get_model(model_type):
    models = {
        "LLM": create_LLM_model,
        "embedding": create_embedding_model
    }

    return models[model_type]()


def create_LLM_model():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto"
    )


def create_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
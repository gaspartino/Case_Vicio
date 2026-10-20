from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def get_model(model_type):
    models = {
        "embedding": create_embedding_model,
        "classifier": create_classifier_model
    }

    return models[model_type]()


def create_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")


def create_classifier_model():
    return {
        "outside_taxonomy": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        ),
        "central_theme": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        )
    }
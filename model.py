from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression

def get_model(model_type):
    models = {"embedding": create_embedding_model, "classifier": create_classifier_model}
    return models[model_type]()

def create_embedding_model():
    return SentenceTransformer("intfloat/multilingual-e5-large-instruct")

def create_classifier_model():
    return {"outside_taxonomy": make_pipeline(Normalizer(norm="l2"), LogisticRegression(max_iter=10000, class_weight="balanced", random_state=42))}
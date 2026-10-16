from transformers import pipeline
from sentence_transformers import SentenceTransformer


def get_model(model_type):
    models = {
        "llm": create_LLM_model,
        "embedding": create_embedding_model
    }

    return models[model_type]()

def create_LLM_model():
    llm = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        clean_up_tokenization_spaces=False,
    )
    gc = llm.generation_config
    gc.max_length = None
    gc.max_new_tokens = 512
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    llm.model.generation_config = gc
    return llm


def create_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

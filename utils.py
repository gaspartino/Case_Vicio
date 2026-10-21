import numpy as np
import re
import json
import html
import faiss
import pandas as pd
import unicodedata
from transformers import pipeline
from json_repair import repair_json
from sentence_transformers import SentenceTransformer, util

def load_questions(file_path, discipline=None, subject=None):
    questions = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            question = json.loads(line)

            if discipline and question.get("disciplina_origem", "").casefold() != discipline.casefold():
                continue

            if subject and subject.casefold() not in question.get("assunto_origem", "").casefold():
                continue

            questions.append(question)

    return questions


def load_themes(csv_path):
    themes = pd.read_csv(csv_path, dtype={"tema_id": str, "parent_id": str})
    required_columns = ["tema_id", "parent_id", "disciplina", "nivel", "titulo", "caminho", "descricao", "status"]
    missing_columns = [column for column in required_columns if column not in themes.columns]

    if missing_columns:
        raise ValueError(f"Colunas ausentes no CSV: {missing_columns}")

    themes["parent_id"] = themes["parent_id"].fillna("").str.strip()
    themes["titulo"] = themes["titulo"].fillna("").str.strip()
    themes["caminho"] = themes["caminho"].fillna("").str.strip()
    themes["descricao"] = themes["descricao"].fillna("").str.strip()
    themes["status"] = themes["status"].fillna("").str.strip()

    if themes["tema_id"].duplicated().any():
        raise ValueError("Existem tema_id duplicados.")

    active_themes = themes[themes["status"].str.lower() == "active"].copy()
    root_themes = active_themes[active_themes["nivel"] == 0].copy()
    classification_themes = active_themes[active_themes["nivel"] > 0].copy()

    return themes, root_themes, classification_themes

def load_training_data(csv_path):
    training_data = pd.read_csv(csv_path, dtype={"id": str, "tema_central_id": str, "extras_ids": str})
    required_columns = ["id", "tema_central_id", "tema_central_caminho", "extras_ids", "fora_da_taxonomia"]
    missing_columns = [column for column in required_columns if column not in training_data.columns]

    if missing_columns:
        raise ValueError(f"Colunas ausentes no CSV: {missing_columns}")

    training_data["id"] = training_data["id"].fillna("").str.strip()
    training_data["tema_central_id"] = training_data["tema_central_id"].fillna("").str.strip()
    training_data["tema_central_caminho"] = training_data["tema_central_caminho"].fillna("").str.strip()
    training_data["extras_ids"] = training_data["extras_ids"].fillna("").str.strip()
    training_data["fora_da_taxonomia"] = pd.to_numeric(training_data["fora_da_taxonomia"], errors="raise").astype(int)

    if training_data["id"].duplicated().any():
        raise ValueError("Existem IDs de questões duplicados.")

    return training_data

def build_complete_training_data(training_data, questions, themes):
    questions_by_id = {str(question.get("id", "")).strip(): question for question in questions}
    themes_by_id = {str(theme["tema_id"]).strip(): theme.to_dict() for _, theme in themes.iterrows()}
    complete_data = []

    for _, row in training_data.iterrows():
        record = row.to_dict()
        question_id = str(record["id"]).strip()
        question = questions_by_id.get(question_id)

        if question is None:
            raise ValueError(f"Questão '{question_id}' não encontrada em questoes.jsonl.")

        record.update({key: value for key, value in question.items() if key != "id"})

        central_id = str(record.get("tema_central_id") or "").strip()
        central_theme = themes_by_id.get(central_id)

        record["tema_central"] = central_theme
        record["tema_central_descricao"] = central_theme.get("descricao", "") if central_theme else ""

        extras_ids = [theme_id.strip() for theme_id in str(record.get("extras_ids") or "").split(";") if theme_id.strip()]
        record["temas_extras"] = [themes_by_id[theme_id] if theme_id in themes_by_id else {"tema_id": theme_id, "nao_encontrado": True} for theme_id in extras_ids]

        complete_data.append(record)

    return pd.DataFrame(complete_data)


def split_questions_by_training_data(questions, training_data):
    training_ids = set(training_data["id"].astype(str).str.strip())

    train_questions = [question for question in questions if str(question.get("id", "")).strip() in training_ids]
    prod_questions = [question for question in questions if str(question.get("id", "")).strip() not in training_ids]

    return train_questions, prod_questions


def build_question_text(question):
    def clean_text(value):
        value = html.unescape(str(value or ""))
        value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
        value = re.sub(r"<[^>]+>", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    statement = clean_text(question.get("enunciado_html", ""))
    command = clean_text(question.get("comando", ""))
    alternatives = question.get("alternativas", {})
    alternatives_text = " ".join(f"{key}: {clean_text(value)}" for key, value in alternatives.items()) if isinstance(alternatives, dict) else clean_text(alternatives)

    return f"Enunciado: {statement}\nComando: {command}\nAlternativas: {alternatives_text}"


def create_theme_index(classification_themes, embedding_model):
    theme_texts = classification_themes.apply(lambda row: f"Tema: {row['titulo']}. Caminho: {row['caminho']}. Descrição: {row['descricao']}", axis=1).tolist()
    embeddings = embedding_model.encode(theme_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def encode_question(question, embedding_model):
    text = build_question_text(question)
    query = f"Instruct: Given a legal question, retrieve the most relevant legal taxonomy theme.\nQuery: {text}"
    return embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)


def retrieve_candidate_themes(question, embedding_model, classification_themes, index, top_k=15):
    text = build_question_text(question)
    query = f"Instruct: Given a legal question, retrieve the most relevant legal taxonomy theme.\nQuery: {text}"
    embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)

    similarities, indices = index.search(embedding, min(top_k, index.ntotal))

    candidates = classification_themes.iloc[indices[0]].copy()
    candidates["similaridade"] = similarities[0]

    return candidates



def train_classifiers(complete_training_data, embedding_model, classifiers):
    data = complete_training_data.copy()
    data["fora_da_taxonomia"] = pd.to_numeric(data["fora_da_taxonomia"], errors="raise").astype(int)

    if not data["fora_da_taxonomia"].isin([0, 1]).all():
        raise ValueError("'fora_da_taxonomia' deve conter apenas 0 ou 1.")

    texts = data.apply(lambda row: build_question_text(row.to_dict()), axis=1).tolist()
    queries = [f"Instruct: Given a legal question, retrieve the most relevant legal taxonomy theme.\nQuery: {text}" for text in texts]
    embeddings = embedding_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    classifiers["outside_taxonomy"].fit(embeddings, data["fora_da_taxonomia"])

    return classifiers


def predict_question(question, embedding_model, classifiers, classification_themes, index, outside_threshold=0.50, theme_threshold=0.75, top_k=15):
    embedding = encode_question(question, embedding_model)

    outside_model = classifiers["outside_taxonomy"]
    outside_probabilities = outside_model.predict_proba(embedding)[0]

    outside_index = list(outside_model.classes_).index(1)
    outside_confidence = float(outside_probabilities[outside_index])

    result = {"id": question.get("id", ""), "decisao": "revisar", "tema_central_id": "", "extras_ids": "", "confianca": outside_confidence}

    if outside_confidence >= outside_threshold:
        result["decisao"] = "fora"
        return result

    candidates = retrieve_candidate_themes(question, embedding_model, classification_themes, index, top_k)

    if candidates.empty:
        return result

    best_candidate = candidates.iloc[0]
    similarity = float(best_candidate["similaridade"])
    result["tema_central_id"] = str(best_candidate["tema_id"])
    result["confianca"] = similarity

    if similarity >= theme_threshold:
        result["decisao"] = "publicar"

    return result
import re
import json
import html
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def create_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


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


def retrieve_candidate_themes(question, classification_themes, model, top_k=5):
    def clean_text(text):
        text = html.unescape(str(text or ""))
        text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    alternatives = question.get("alternativas", {})
    alternatives_text = " ".join(f"{key}: {value}" for key, value in alternatives.items()) if isinstance(alternatives, dict) else ""

    question_text = " ".join(filter(None, [clean_text(question.get("enunciado_html", "")), clean_text(question.get("comando", "")), clean_text(alternatives_text), clean_text(question.get("disciplina_origem", "")), clean_text(question.get("assunto_origem", ""))]))

    if not question_text:
        return pd.DataFrame()

    themes = classification_themes.copy()
    text_columns = ["disciplina", "titulo", "caminho", "descricao"]

    for column in text_columns:
        themes[column] = themes[column].fillna("").astype(str).str.strip()

    discipline = clean_text(question.get("disciplina_origem", "")).casefold()

    if discipline:
        matching_themes = themes[themes["disciplina"].str.casefold() == discipline]
        if not matching_themes.empty:
            themes = matching_themes

    if themes.empty:
        return pd.DataFrame()

    themes["texto_busca"] = themes.apply(lambda row: f"Título: {row['titulo']}. Caminho: {row['caminho']}. Descrição: {row['descricao']}", axis=1)

    question_embedding = model.encode(question_text, convert_to_tensor=True, normalize_embeddings=True)
    theme_embeddings = model.encode(themes["texto_busca"].tolist(), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

    themes["similaridade"] = util.cos_sim(question_embedding, theme_embeddings)[0].cpu().numpy()

    return themes.sort_values("similaridade", ascending=False).head(top_k).drop(columns=["texto_busca"]).reset_index(drop=True)


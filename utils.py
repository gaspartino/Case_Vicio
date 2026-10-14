import re
import json
import html
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

def get_model(model_type):
    models = {
        "llm": create_LLM_model,
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

def classify_question(question, candidates, llm):
    alternatives = candidates[["tema_id", "titulo", "caminho", "descricao", "similaridade"]].fillna("").to_dict(orient="records")
    question_text = json.dumps(question, ensure_ascii=False)
    candidates_text = json.dumps(alternatives, ensure_ascii=False)

    prompt = f"""
      Você é um classificador de questões jurídicas para uma plataforma de estudos.

      Analise a questão e os temas candidatos. Escolha um tema somente se ele corresponder ao conteúdo jurídico efetivamente cobrado.

      Regras:
      - Considere o enunciado, o comando e as alternativas.
      - Use disciplina_origem e assunto_origem como pistas, não como verdade absoluta.
      - Considere o título, a descrição e o caminho hierárquico de cada tema.
      - Se um tema candidato for adequado, retorne seu tema_id.
      - Use "publicar" somente quando houver alta confiança na correspondência.
      - Use "revisar" quando houver um candidato plausível, mas persistir dúvida.
      - Use "fora" quando nenhum candidato for adequado.
      - Não invente IDs de temas.
      - A similaridade é apenas uma evidência auxiliar, não uma probabilidade.
      - Retorne somente JSON válido, sem Markdown.

      Questão:
      {question_text}

      Temas candidatos:
      {candidates_text}

      Formato obrigatório:
      {{
        "tema_id": null,
        "decisao": "revisar",
        "confianca": 0.0,
        "justificativa": "Explicação objetiva da decisão."
      }}
    """

    response = llm(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
    content = response[0]["generated_text"].strip()
    result = json.loads(content)

    valid_ids = {str(candidate["tema_id"]) for candidate in alternatives if candidate["tema_id"] != ""}
    selected_id = result.get("tema_id")
    if selected_id is not None and str(selected_id) not in valid_ids:
        raise ValueError(f"O LLM retornou um tema_id que não está entre os candidatos: {selected_id}")
    if result.get("decisao") not in {"publicar", "revisar", "fora"}:
        raise ValueError("Decisão inválida retornada pelo LLM.")
    if result["decisao"] == "fora":
        result["tema_id"] = None
    elif selected_id is None:
        raise ValueError("É necessário informar tema_id para decisões publicar ou revisar.")
    if not isinstance(result.get("confianca"), (int, float)) or not 0 <= result["confianca"] <= 1:
        raise ValueError("A confiança deve ser um número entre 0 e 1.")

    return result



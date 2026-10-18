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


def normalize_discipline(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"\s+", " ", value).strip().casefold()
    aliases = {"direito penal": "penal", "penal": "penal", "direito processual penal": "processo penal", "processo penal": "processo penal", "criminologia": "criminologia", "legislacao penal especial": "legislacao penal especial"}
    return aliases.get(value, value)
    


def normalize_discipline(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"\s+", " ", value).strip().casefold()
    aliases = {"direito penal": "penal", "penal": "penal", "direito processual penal": "processo penal", "processo penal": "processo penal", "criminologia": "criminologia", "legislacao penal especial": "legislacao penal especial"}
    return aliases.get(value, value)


def build_theme_index(classification_themes, model):
    themes = classification_themes.copy().reset_index(drop=True)
    for column in ["disciplina", "titulo", "caminho", "descricao"]:
        themes[column] = themes[column].fillna("").astype(str).str.strip()
    themes["disciplina_norm"] = themes["disciplina"].apply(normalize_discipline)
    themes["texto_busca"] = themes.apply(lambda row: f"Disciplina: {row['disciplina']}. Tema: {row['titulo']}. Hierarquia temática: {row['caminho']}. Descrição: {row['descricao']}", axis=1)
    embeddings = model.encode(themes["texto_busca"].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    global_index = faiss.IndexFlatIP(embeddings.shape[1])
    global_index.add(embeddings)
    indexes = {"global": (global_index, themes)}
    for discipline in themes["disciplina_norm"].unique():
        subset = themes[themes["disciplina_norm"] == discipline].copy().reset_index(drop=True)
        if subset.empty:
            continue
        subset_embeddings = model.encode(subset["texto_busca"].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        subset_embeddings = np.ascontiguousarray(subset_embeddings, dtype=np.float32)
        discipline_index = faiss.IndexFlatIP(subset_embeddings.shape[1])
        discipline_index.add(subset_embeddings)
        indexes[discipline] = (discipline_index, subset)
    return indexes, themes


def retrieve_candidate_themes(question, classification_themes, model, index, top_k=15):
    def clean_text(text):
        text = html.unescape(str(text or ""))
        text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    alternatives = question.get("alternativas", {})
    alternatives_text = " ".join(f"{key}: {value}" for key, value in alternatives.items()) if isinstance(alternatives, dict) else ""
    question_text = " ".join(filter(None, [clean_text(question.get("enunciado_html", "")), clean_text(question.get("comando", "")), clean_text(alternatives_text), clean_text(question.get("assunto_origem", ""))]))
    if not question_text:
        return pd.DataFrame()
    discipline = normalize_discipline(question.get("disciplina_origem", ""))
    search_index, themes = index.get(discipline, index["global"])
    question_embedding = model.encode([question_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    question_embedding = np.ascontiguousarray(question_embedding, dtype=np.float32)
    scores, indices = search_index.search(question_embedding, min(top_k, len(themes)))
    valid = indices[0] >= 0
    result = themes.iloc[indices[0][valid]].copy()
    result["similaridade"] = scores[0][valid]
    return result.sort_values("similaridade", ascending=False).reset_index(drop=True).drop(columns=["texto_busca", "disciplina_norm"], errors="ignore")



def classify_question(question, candidates, llm):
    alternatives = candidates[["tema_id", "titulo", "caminho", "descricao", "similaridade"]].fillna("").to_dict(orient="records")
    if not alternatives:
        raise ValueError("Nenhum tema candidato foi fornecido.")

    valid_ids = {str(candidate["tema_id"]).strip() for candidate in alternatives if str(candidate["tema_id"]).strip()}
    question_text = json.dumps(question, ensure_ascii=False)
    candidates_text = json.dumps(alternatives, ensure_ascii=False)

    prompt = f"""Você é um classificador de questões jurídicas para uma plataforma de estudos.

Sua tarefa é identificar o tema que melhor corresponde ao conteúdo jurídico efetivamente cobrado na questão.

REGRAS DE CLASSIFICAÇÃO:
- Analise o enunciado, o comando e todas as alternativas.
- Use disciplina_origem e assunto_origem como pistas auxiliares, não como verdade absoluta.
- Considere o título, a descrição e o caminho hierárquico de cada tema candidato.
- Identifique o instituto jurídico central efetivamente cobrado.
- Não escolha um tema apenas porque sua similaridade é a maior.
- Não invente relações jurídicas para justificar uma escolha.
- "publicar": correspondência temática forte e direta.
- "revisar": existe um candidato plausível, mas há incerteza.
- "fora": nenhum candidato corresponde adequadamente.

REGRAS OBRIGATÓRIAS PARA tema_id:
- Para "publicar" ou "revisar", selecione o ID de um dos candidatos.
- Para "fora", retorne tema_id null.
- Copie o tema_id exatamente como aparece na lista.
- Nunca invente IDs nem selecione temas ausentes dos candidatos.

REGRAS PARA A CONFIANÇA:
- Use um número decimal entre 0 e 1, inclusive.
- Não retorne porcentagens como 85 ou 95.
- A confiança deve refletir a força das evidências temáticas.
- A confiança é uma estimativa subjetiva, não uma probabilidade calibrada.
- Não use a confiança isoladamente para decidir a publicação.

REGRAS DE SAÍDA:
- Retorne exclusivamente um objeto JSON válido, sem Markdown ou texto adicional.
- Use exatamente as chaves "tema_id", "decisao", "confianca" e "justificativa".
- "decisao" deve ser "publicar", "revisar" ou "fora".
- "justificativa" deve explicar objetivamente a escolha.
- Não utilize aspas duplas sem escape dentro de valores de texto.

Formato obrigatório:
{{"tema_id": "ID_DO_CANDIDATO_OU_NULL", "decisao": "publicar", "confianca": 0.85, "justificativa": "Correspondência temática identificada."}}

Questão:
{question_text}

Temas candidatos:
{candidates_text}
"""

    last_error = None

    for attempt in range(2):
        current_prompt = prompt if attempt == 0 else f"""{prompt}

ATENÇÃO: A resposta anterior não passou pela validação.
Motivo: {last_error}

Gere uma nova resposta corrigindo o problema. Retorne um único objeto JSON válido, com as quatro chaves exigidas, um ID existente para "publicar" ou "revisar", null para "fora" e confiança numérica entre 0 e 1.
"""

        try:
            response = llm([{"role": "user", "content": current_prompt}], return_full_text=False)
            content = response[0]["generated_text"].strip()

            if not content:
                raise ValueError("O LLM retornou uma resposta vazia.")

            match = re.search(r"\{[\s\S]*\}", content)

            if not match:
                raise ValueError(f"O LLM não retornou um objeto JSON: {content!r}")

            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                result = repair_json(match.group(0), return_objects=True)

            if not isinstance(result, dict):
                raise ValueError("A resposta não é um objeto JSON.")

            if "decisao" not in result and "decisão" in result:
                result["decisao"] = result.pop("decisão")

            if "confianca" not in result and "confiança" in result:
                result["confianca"] = result.pop("confiança")

            decisao = result.get("decisao")

            if isinstance(decisao, str):
                decisao = decisao.strip().lower()
                decisao = {"publlicar": "publicar"}.get(decisao, decisao)

            if decisao not in {"publicar", "revisar", "fora"}:
                raise ValueError(f"Decisão inválida: {decisao!r}")

            selected_id = result.get("tema_id")

            if decisao == "fora":
                result["tema_id"] = None
            else:
                if selected_id is None or str(selected_id).strip() not in valid_ids:
                    raise ValueError(f"ID de tema ausente ou inválido: {selected_id!r}")

                result["tema_id"] = next(candidate["tema_id"] for candidate in alternatives if str(candidate["tema_id"]).strip() == str(selected_id).strip())

            confianca = result.get("confianca")

            if isinstance(confianca, str):
                try:
                    confianca = float(confianca.strip().replace(",", "."))
                except ValueError:
                    raise ValueError(f"Confiança não numérica: {confianca!r}")

            if not isinstance(confianca, (int, float)) or isinstance(confianca, bool) or not 0 <= confianca <= 1:
                raise ValueError(f"Confiança deve estar entre 0 e 1: {confianca!r}")

            justificativa = result.get("justificativa")

            if not isinstance(justificativa, str) or not justificativa.strip():
                raise ValueError("Justificativa ausente ou inválida.")

            result["decisao"] = decisao
            result["confianca"] = float(confianca)
            result["justificativa"] = justificativa.strip()

            return result

        except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            print(f"Tentativa {attempt + 1}/2 falhou: {last_error}")

    raise ValueError(f"Falha após 2 tentativas de classificação: {last_error}")

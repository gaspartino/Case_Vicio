import re
import json
import html
import pandas as pd
from transformers import pipeline
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
    prompt = f"""Você é um classificador de questões jurídicas para uma plataforma de estudos.

Sua tarefa é identificar o tema que melhor corresponde ao conteúdo jurídico efetivamente cobrado na questão.

REGRAS DE CLASSIFICAÇÃO:
- Analise o enunciado, o comando e todas as alternativas disponíveis.
- Use disciplina_origem e assunto_origem como pistas auxiliares, não como verdade absoluta.
- Considere o título, a descrição e o caminho hierárquico de cada tema candidato.
- Escolha o tema com a melhor correspondência ao conteúdo jurídico central da questão.
- Não escolha um tema apenas porque sua similaridade é a maior.
- Use "publicar" somente quando houver forte correspondência temática.
- Use "revisar" quando existir um candidato plausível, mas houver incerteza sobre a correspondência.
- Use "fora" somente quando nenhum dos candidatos for adequado.

REGRAS OBRIGATÓRIAS PARA tema_id:
- Para "publicar", selecione o ID de um dos candidatos.
- Para "revisar", selecione obrigatoriamente o ID do candidato mais plausível, mesmo que exista dúvida.
- Nunca retorne tema_id null para "publicar" ou "revisar".
- Para "fora", retorne tema_id null.
- Copie o tema_id exatamente como aparece na lista de candidatos.
- Não invente IDs nem selecione temas ausentes da lista.

REGRAS ADICIONAIS PARA A CONFIANÇA:
- Estime a confiança de acordo com as evidências específicas da questão e dos temas candidatos.
- Não copie valores de confiança de exemplos ou respostas anteriores.
- Utilize valores diferentes quando a força das evidências for diferente.
- A confiança deve refletir a segurança da escolha do tema, não apenas a similaridade textual.
- Não use a confiança isoladamente para decidir se uma questão será publicada.

REGRAS DE SAÍDA:
- Retorne exclusivamente um objeto JSON válido, sem Markdown ou texto adicional.
- Use exatamente as chaves "tema_id", "decisao", "confianca" e "justificativa".
- "decisao" deve ser "publicar", "revisar" ou "fora".
- "confianca" deve ser um número entre 0 e 1, representando a confiança estimada pelo modelo, não uma probabilidade calibrada.
- "justificativa" deve explicar objetivamente a correspondência temática ou a ausência dela.
- A similaridade é apenas uma evidência auxiliar, não uma probabilidade.

Questão:
{question_text}

Temas candidatos:
{candidates_text}

"""

    response = llm(
        [{"role": "user", "content": prompt}],
        return_full_text=False,
    )
    content = response[0]["generated_text"].strip()
    print("Resposta do Qwen:", repr(content))
    if not content:
        raise ValueError("O LLM retornou uma resposta vazia.")
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        raise ValueError(f"O LLM não retornou JSON. Resposta recebida: {content!r}")
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"O LLM retornou JSON inválido: {content!r}") from exc

    if "decisao" not in result and "decisão" in result:
        result["decisao"] = result.pop("decisão")
    if "confianca" not in result and "confiança" in result:
        result["confianca"] = result.pop("confiança")

    valid_ids = {str(candidate["tema_id"]) for candidate in alternatives if candidate["tema_id"] != ""}
    selected_id = result.get("tema_id")
    decisao = result.get("decisao")
    if isinstance(decisao, str):
        decisao = decisao.strip().lower()
    result["decisao"] = decisao
    if decisao not in {"publicar", "revisar", "fora"}:
        raise ValueError(f"Decisão inválida retornada pelo LLM: {content!r}")
    if decisao == "fora":
        result["tema_id"] = None
    elif selected_id is None or str(selected_id) not in valid_ids:
        raise ValueError(f"O LLM selecionou um tema ausente dos candidatos: {selected_id}")
    elif selected_id is None or str(selected_id) not in valid_ids:
        raise ValueError(f"O LLM selecionou um tema ausente dos candidatos: {selected_id}")
    confianca = result.get("confianca")
    if not isinstance(confianca, (int, float)) or isinstance(confianca, bool) or not 0 <= confianca <= 1:
        raise ValueError("A confiança deve ser um número entre 0 e 1.")
    if not isinstance(result.get("justificativa"), str):
        raise ValueError("A justificativa retornada pelo LLM deve ser texto.")
    if decisao != "fora":
        result["tema_id"] = next(candidate["tema_id"] for candidate in alternatives if str(candidate["tema_id"]) == str(selected_id))
    return result

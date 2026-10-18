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
    question_text = " ".join(filter(None, [clean_text(question.get("enunciado_html", "")), clean_text(question.get("comando", "")), clean_text(alternatives_text)]))
    if not question_text:
        return pd.DataFrame()
    search_index, themes = index["global"]
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

REGRAS DE CLASSIFICAÇÃO E DECISÃO:

Analise integralmente o enunciado, o comando e todas as alternativas da questão. Utilize `disciplina_origem` e `assunto_origem` como informações auxiliares, sem considerá-las necessariamente corretas. Compare o conteúdo da questão com o título, a descrição e o caminho hierárquico dos temas candidatos, identificando o instituto jurídico central efetivamente cobrado.

Não escolha um tema apenas por apresentar a maior similaridade semântica. A decisão deve ser fundamentada na correspondência jurídica e temática entre a questão e o tema escolhido. Não invente relações jurídicas, não force uma correspondência e não confunda assuntos relacionados com equivalência temática.

1. publicar — correspondência temática suficientemente segura**

Escolha `publicar` quando houver evidências claras de que o tema selecionado corresponde ao conteúdo efetivamente cobrado e quando a confiança na classificação for suficientemente alta para justificar a publicação automática.

A correspondência deve ser avaliada considerando a hierarquia da árvore temática. Uma classificação pode ser considerada correta quando o tema escolhido for o mesmo tema atribuído pela equipe ou um ancestral ou descendente pertencente ao mesmo ramo, conforme a regra de avaliação do desafio. Entretanto, não considere automaticamente corretos dois subtemas diferentes apenas por pertencerem ao mesmo tema pai.

Como uma publicação incorreta faz o aluno estudar a questão no assunto errado, adote um critério conservador: publique somente quando houver evidências suficientes para sustentar a escolha. O objetivo é alcançar pelo menos 95% de acerto entre as questões publicadas e, respeitada essa condição, maximizar a quantidade de questões publicadas.

2. revisar — correspondência plausível, mas insuficientemente segura**

Escolha `revisar` quando existir um tema candidato razoável, mas houver incerteza relevante que impeça a publicação automática.

Isso inclui situações em que:
- Dois ou mais temas apresentam correspondência jurídica plausível.
- O tema candidato trata de assunto relacionado, mas não está claro se representa o instituto jurídico central da questão.
- O título, a descrição ou a posição hierárquica dos candidatos não permitem distinguir com segurança o tema adequado.
- As informações de origem entram em conflito com o conteúdo efetivo da questão.
- Há indícios de que o tema correto não foi recuperado entre os candidatos disponíveis, mas ainda existe uma hipótese plausível que merece avaliação humana.

Sempre que possível, indique o tema mais provável como sugestão e explique objetivamente a dúvida que exige intervenção do professor. A revisão deve ser utilizada para resolver incertezas reais, evitando encaminhar questões que poderiam ser classificadas com segurança.

3. fora — ausência de correspondência adequada**

Escolha `fora` quando nenhum dos temas candidatos representar adequadamente o instituto jurídico central cobrado pela questão.

Utilize essa decisão quando os candidatos disponíveis forem apenas superficialmente relacionados ao enunciado, pertencerem a assuntos juridicamente distintos ou não houver evidências suficientes de que algum deles seja uma opção temática válida.

Não escolha um tema inadequado apenas para evitar essa decisão. Por outro lado, não use `fora` simplesmente porque a classificação é difícil: se houver um candidato plausível, mas incerto, prefira `revisar`.

**CRITÉRIO GERAL DE PRIORIZAÇÃO**

A prioridade é maximizar o número de questões publicadas sem reduzir a precisão das publicações abaixo de 95%. Portanto, a decisão não deve buscar publicar o maior número possível de questões a qualquer custo. Questões com correspondência forte e direta devem ser publicadas; questões plausíveis, mas incertas, devem ser revisadas; e questões sem correspondência temática adequada devem ser classificadas como fora.

A justificativa de cada decisão deve explicar a relação entre o instituto jurídico cobrado e o tema escolhido, indicando por que a correspondência é segura, por que exige revisão ou por que nenhum candidato é adequado.

REGRAS OBRIGATÓRIAS PARA tema_id:
- Para "publicar" ou "revisar", selecione o ID de um dos candidatos.
- Para "fora", retorne tema_id null.
- Copie o tema_id exatamente como aparece na lista.
- Nunca invente IDs nem selecione temas ausentes dos candidatos.

REGRAS PARA A CONFIANÇA:

- Retorne um número decimal entre 0 e 1, sem se limitar a valores predefinidos.
- Avalie a confiança individualmente para cada questão, com base nas evidências disponíveis. Não reutilize automaticamente valores atribuídos a questões anteriores.
- **0.90–1.00:** correspondência temática direta, instituto jurídico claramente identificado e pouca ambiguidade entre os candidatos.
- **0.75–0.89:** correspondência forte, mas com alguma dúvida sobre a especificidade do tema ou sobre a distinção entre candidatos próximos.
- **0.50–0.74:** correspondência plausível, com incertezas relevantes ou evidências incompletas.
- **0.25–0.49:** correspondência fraca, sustentada principalmente por relações indiretas ou semelhanças superficiais.
- **0.00–0.24:** evidências mínimas de correspondência com o tema escolhido.
- As faixas são referências qualitativas, não probabilidades calibradas nem limites automáticos para publicação.
- Não escolha uma faixa apenas para justificar a decisão tomada. Primeiro avalie a correspondência temática; depois atribua a confiança compatível com as evidências.
- Se nenhum candidato for adequado, não atribua confiança alta à correspondência com um tema específico. Diferencie a confiança na escolha de um tema da confiança de que nenhum candidato serve.
- A justificativa deve apresentar evidências compatíveis com a confiança atribuída.

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

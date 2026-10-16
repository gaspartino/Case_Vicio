import json
import pandas as pd
from utils import load_questions, load_themes, retrieve_candidate_themes, classify_question
from model import get_model

def main():
    questions = load_questions("data/questoes.jsonl")
    themes, root_themes, classification_themes = load_themes("data/temas.csv")
    embedding = get_model("embedding")
    llm = get_model("llm")
    results = []
    candidate_results = []
    output_path = "publicacao.csv"

    for index, question in enumerate(questions, start=1):
        question_id = question.get("id", f"Q{index:04d}")
        print(f"[{index}/{len(questions)}] Classificando {question_id}")

        try:
            candidates = retrieve_candidate_themes(question, classification_themes, embedding, top_k=5)

            for candidate in candidates.to_dict(orient="records"):
                candidate_results.append({"questao_id": question_id, **candidate})

            classification = classify_question(question, candidates, llm)
            selected_id = classification.get("tema_id")
            selected_theme = candidates[candidates["tema_id"].astype(str) == str(selected_id)] if selected_id is not None else pd.DataFrame()
            theme_title = selected_theme.iloc[0]["titulo"] if not selected_theme.empty else None

            question_data = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in question.items()}
            results.append({**question_data, **classification, "tema_titulo": theme_title, "status_processamento": "sucesso", "erro": None})

        except Exception as exc:
            question_data = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in question.items()}
            results.append({**question_data, "tema_id": None, "tema_titulo": None, "decisao": "erro", "confianca": None, "justificativa": None, "status_processamento": "erro", "erro": str(exc)})
            print(f"Erro na questão {question_id}: {exc}")

        if index % 10 == 0 or index == len(questions):
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Progresso salvo: {index}/{len(questions)} questões")

    print(f"Processamento concluído. Arquivo salvo em: {output_path}")
    print(pd.DataFrame(results)["decisao"].value_counts(dropna=False))

if __name__ == "__main__":
    main()

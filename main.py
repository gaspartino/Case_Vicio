from utils import get_model, load_questions, load_themes, retrieve_candidate_themes, classify_question

def main():
    embedding = get_model("embedding")
    llm = get_model("llm")

    questions = load_questions("data/questoes.jsonl")
    themes, root_themes, classification_themes = load_themes("data/temas.csv")

    question = questions[1]
    candidates = retrieve_candidate_themes(question, classification_themes, embedding, top_k=5)
    result = classify_question(question, candidates, llm)

    print(result)

    
if __name__ == "__main__":
    main()
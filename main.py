from utils import create_embedding_model, load_questions, load_themes, retrieve_candidate_themes


def main():
    model = create_embedding_model()
    questions = load_questions("data/questoes.jsonl")
    themes, root_themes, classification_themes = load_themes("data/temas.csv")

    candidates = retrieve_candidate_themes(questions[1], classification_themes, model, top_k=5)

    print(candidates[["tema_id", "titulo", "caminho", "similaridade"]])

    
if __name__ == "__main__":
    main()
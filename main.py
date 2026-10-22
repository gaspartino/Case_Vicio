import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import get_model
from utils import load_questions, load_themes, load_training_data, build_complete_training_data, train_classifiers, predict_question, split_questions_by_training_data, create_theme_index, encode_question


def main(test=False):
    training_data = load_training_data("data/folha_verdade.csv")
    questions = load_questions("data/questoes.jsonl")
    themes, root_themes, classification_themes = load_themes("data/temas.csv")

    train_questions, prod_questions = split_questions_by_training_data(questions, training_data)

    embedding_model = get_model("embedding")
    classifiers = get_model("classifier")
    theme_index, theme_embeddings = create_theme_index(classification_themes, embedding_model)

    output_path = "publicacao.csv"

    if test:
        labels_by_id = training_data.set_index("id")["fora_da_taxonomia"]
        labels = [int(labels_by_id.loc[question["id"]]) for question in train_questions]

        train_questions, test_questions = train_test_split(train_questions, test_size=0.05, stratify=labels)

        train_ids = {str(question["id"]) for question in train_questions}
        test_ids = {str(question["id"]) for question in test_questions}

        train_labels = training_data[training_data["id"].isin(train_ids)]
        test_labels = training_data[training_data["id"].isin(test_ids)]

        complete_training_data = build_complete_training_data(train_labels, train_questions, themes)
        classifiers = train_classifiers(complete_training_data, embedding_model, classifiers)

        predictions = [
            predict_question(question, encode_question(question, embedding_model), classifiers, classification_themes, theme_index)
            for question in test_questions
        ]
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"Previsões geradas: {len(predictions_df)}")
        print(f"Arquivo salvo em {output_path}")

    else:
        complete_training_data = build_complete_training_data(training_data, questions, themes)
        classifiers = train_classifiers(complete_training_data, embedding_model, classifiers)

        predictions = [
            predict_question(question, encode_question(question, embedding_model), classifiers, classification_themes, theme_index)
            for question in prod_questions
        ]

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"Previsões geradas: {len(predictions_df)}")
        print(f"Arquivo salvo em {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args.test)
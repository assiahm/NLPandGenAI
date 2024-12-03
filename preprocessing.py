import json
import os
import spacy


##1chargement des données
def load_data(file_path):
    """Charge un fichier JSON Lines et retourne une liste de dictionnaires."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def extract_fields(data, fields):
    """Extrait les champs pertinents de chaque entrée."""
    extracted_data = [{field: item.get(field, "") for field in fields} for item in data]
    return extracted_data

if __name__ == "__main__":
    # Chemins vers les fichiers
    reviews_path = "data/reviews.jsonl"


    # Chargement des données
    reviews = load_data(reviews_path)

    # Sélection des champs "title" et "text" dans les reviews
    selected_reviews = extract_fields(reviews, ["title", "text"])
    print(f"Nombre d'avis chargés : {len(selected_reviews)}")


##2 traitement linguistique

# Charger le modèle de langue anglaise
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Prétraite un texte : tokenisation, lemmatisation, suppression des stop words et bruit."""
    doc = nlp(text.lower())  # Convertir en minuscule et passer dans le modèle spaCy
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return tokens

def preprocess_reviews(reviews):
    """Applique le prétraitement à une liste d'avis."""
    for review in reviews:
        review['tokens'] = preprocess_text(review.get('text', ""))
    return reviews

if __name__ == "__main__":
    # Appliquer le prétraitement sur les données sélectionnées
    processed_reviews = preprocess_reviews(selected_reviews)
    print(f"Exemple de tokens : {processed_reviews[0]['tokens']}")

##Sauvegarde des données préparés

def save_preprocessed_data(data, output_file):
    """Sauvegarde les données dans un fichier JSON."""
    output_path = os.path.join("data", output_file)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    print(f"Données sauvegardées dans : {output_path}")

if __name__ == "__main__":
    # Sauvegarder les données prétraitées
    save_preprocessed_data(processed_reviews, "preprocessed_reviews.json")



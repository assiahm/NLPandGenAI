import json
import os

def load_reviews(file_name):
    """Charge les avis clients depuis un fichier JSON."""
    input_path = os.path.join("data", file_name)
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_text_and_ratings(data):
    """Extrait les textes et les notes réelles (ratings) des avis."""
    texts = [item['text'] for item in data if 'text' in item]
    ratings = [item.get('rating', None) for item in data if 'rating' in item]
    return texts, ratings

if __name__ == "__main__":
    # Charger les données
    reviews = load_reviews("preprocessed_reviews.json")
    texts, ratings = extract_text_and_ratings(reviews)
    print(f"Nombre d'avis : {len(texts)}")
    print(f"Exemple de texte : {texts[0]}")
    print(f"Exemple de rating : {ratings[0]}")

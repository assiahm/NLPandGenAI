import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


#1 charger le fichier Json contenant les tokens prétraités
##Commencez par charger le fichier preprocessed_reviews.json contenant les tokens nettoyés créés dans l'étape précédente.
def load_preprocessed_data(file_name):
    """Charge les données prétraitées depuis un fichier JSON."""
    input_path = os.path.join("data", file_name)
    with open(input_path, 'r', encoding='utf-8') as file:
        return json.load(file)

if __name__ == "__main__":
    # Charger les données
    preprocessed_reviews = load_preprocessed_data("preprocessed_reviews.json")
    print(f"Nombre de documents chargés : {len(preprocessed_reviews)}")
    print(f"Exemple de document : {preprocessed_reviews[0]}")


#///////////////////////////////////////


#2 Générer des embeddings(transforment chaque document en représentation vectorielle )
#Les embeddings transforment chaque document en une représentation vectorielle. Utilisons le modèle all-MiniLM-L6-v2 via sentence-transformers.
def generate_embeddings(reviews, model_name="all-MiniLM-L6-v2"):
    """Génère des embeddings pour une liste de documents."""
    model = SentenceTransformer(model_name)
    documents = [" ".join(review['tokens']) for review in reviews]
    embeddings = model.encode(documents)
    return embeddings

if __name__ == "__main__":
    # Génération des embeddings
    embeddings = generate_embeddings(preprocessed_reviews)
    print(f"Nombre d'embeddings générés : {len(embeddings)}")
    print(f"Exemple d'embedding : {embeddings[0][:10]}")  # Affiche les 10 premières dimensions

#//////////////////////////////////////////////////////////////////////
#3 Appliquer le clustering  KMEANS
#Utilisons KMeans pour regrouper les documents en clusters. Choisissez un nombre de clusters (num_clusters) en fonction des données.
def cluster_documents(embeddings, num_clusters=10):## 5 CLUSTERS
    """Applique le clustering KMeans aux embeddings."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f"///////Silhouette Score : {score}")
    return labels, kmeans

if __name__ == "__main__":
    # Appliquer KMeans
    num_clusters = 3
    labels, kmeans_model = cluster_documents(embeddings, num_clusters)
    print(f"Exemple de labels : {labels[:10]}")

#///////////////////////////////////////////////////////////////////
#Analyse des clusters


def analyze_clusters(reviews, labels):
    """Analyse les clusters en regroupant les documents et en comptant les mots fréquents."""
    clusters = {i: [] for i in set(labels)}
    for review, label in zip(reviews, labels):
        clusters[label].extend(review['tokens'])
    
    # Calcul des mots fréquents
    cluster_keywords = {}
    for cluster_id, tokens in clusters.items():
        word_counts = Counter(tokens)
        most_common = word_counts.most_common(10)
        cluster_keywords[cluster_id] = most_common
        print(f"Cluster {cluster_id} - Mots fréquents : {most_common}\n")
    
    return clusters, cluster_keywords

if __name__ == "__main__":
    # Analyser les clusters
    clusters, cluster_keywords = analyze_clusters(preprocessed_reviews, labels)


##choix clusteringimport matplotlib.pyplot as plt
def evaluate_clusters(embeddings, max_clusters=10):
    """Évalue la qualité des clusters avec l'inertie et le Silhouette Score."""
    inertias = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculer l'inertie
        inertias.append(kmeans.inertia_)
        
        # Calculer le Silhouette Score
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
    
    # Afficher les résultats
    plt.figure(figsize=(14, 6))

    # Courbe de l'inertie
    plt.subplot(1, 2, 1)
    plt.plot(cluster_range, inertias, marker='o')
    plt.title("Méthode Elbow (Inertie)")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.grid(True)

    # Courbe du Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title("Silhouette Score")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
evaluate_clusters(embeddings, max_clusters=10)

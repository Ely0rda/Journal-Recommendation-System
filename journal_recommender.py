import json
import torch
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk

# Télécharger les données nécessaires pour NLTK (mots vides et lemmatisation)
nltk.download('stopwords')
nltk.download('wordnet')

# Monter Google Drive pour accéder aux fichiers
from google.colab import drive
drive.mount('/content/drive')

class SimpleJournalRecommender:
    def __init__(self, min_samples=2):
        # Initialisation des paramètres de base
        self.min_samples = min_samples  # Nombre minimum d'échantillons pour un journal
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Utiliser GPU si disponible
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # Tokenizer pour DistilBERT
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)  # Modèle DistilBERT
        self.lemmatizer = WordNetLemmatizer()  # Lemmatizer pour réduire les mots à leur racine
        self.stop_words = set(stopwords.words('english'))  # Mots vides en anglais

    def load_data(self, filepath):
        # Charger les données depuis un fichier JSON
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Préparer les données pour le DataFrame
        records = []
        for entry in data:
            records.append({
                'text': f"{entry['Title']} {entry['Abstract']} {' '.join(entry['Keywords'])}",  # Combiner titre, résumé et mots-clés
                'journal_name': entry['Journal']['journal_name'],  # Nom du journal
                'journal_score': entry['Journal']['Journal_Score']  # Score du journal
            })

        # Créer un DataFrame et filtrer les journaux avec un nombre d'échantillons suffisant
        df = pd.DataFrame(records)
        counts = df['journal_name'].value_counts()
        valid_journals = counts[counts >= self.min_samples].index
        self.df = df[df['journal_name'].isin(valid_journals)].reset_index(drop=True)

    def preprocess(self, text):
        # Prétraitement du texte : minuscules, suppression des caractères spéciaux, lemmatisation
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)  # Supprimer les caractères non alphabétiques
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]  # Lemmatiser et enlever les mots vides
        return ' '.join(tokens)

    def get_embedding(self, text):
        # Obtenir l'embedding du texte avec DistilBERT
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text,
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=512,
                                  padding=True).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Retourner l'embedding du premier token

    def recommend(self, title, abstract, keywords=None, top_k=5):
        # Combiner et prétraiter la requête
        query_text = f"{title} {abstract}"
        if keywords:
            query_text += f" {' '.join(keywords)}"
        query_text = self.preprocess(query_text)

        # Obtenir l'embedding de la requête
        query_embed = self.get_embedding(query_text)

        # Générer les embeddings pour le dataset si pas déjà fait
        if not hasattr(self, 'embeddings'):
            print("Génération des embeddings pour le dataset...")
            texts = [self.preprocess(text) for text in self.df['text']]
            self.embeddings = np.vstack([self.get_embedding(text) for text in texts])

        # Calculer les similarités cosinus entre la requête et les embeddings du dataset
        similarities = cosine_similarity(query_embed, self.embeddings).flatten()

        # Obtenir les indices des top_k recommandations
        top_indices = similarities.argsort()[-top_k*3:][::-1]
        recommendations = []
        seen_journals = set()

        # Récupérer les recommandations en évitant les doublons
        for idx in top_indices:
            jname = self.df.iloc[idx]['journal_name']
            if jname not in seen_journals:
                seen_journals.add(jname)
                recommendations.append({
                    'journal_name': jname,
                    'similarity': similarities[idx],
                    'journal_score': self.df.iloc[idx]['journal_score']
                })
            if len(recommendations) >= top_k:
                break

        # Trier les recommandations par score de journal
        recommendations.sort(key=lambda x: x['journal_score'], reverse=True)
        return recommendations[:top_k]

# Exemple d'utilisation
if __name__ == "__main__":
    recommender = SimpleJournalRecommender(min_samples=2)
    recommender.load_data('/content/drive/MyDrive/processed_data.json')  # Charger les données

    # Définir un titre, un résumé et des mots-clés pour tester
    test_title = "Recognition of Arabic handwritten words using convolutional neural network"
    test_abstract = "A new method for recognizing automatically Arabic handwritten words was presented using convolutional neural network architecture. The proposed method is based on global approaches, which consists of recognizing all the words without segmenting into the characters in order to recognize them separately. Convolutional neural network (CNN) is a particular supervised type of neural network based on multilayer principle; our method needs a big dataset of word images to obtain the best result. To optimize our system, a new database was collected from the benchmarking Arabic handwriting database using the pre-processing such as rotation transformation, which is applied on the images of the database to create new images with different features. The convolutional neural network applied on our database that contains 40,320 of Arabic handwritten words (26,880 images for training set and 13,440 for test set). Thus, different configurations on a public benchmark database were evaluated and compared with previous methods. Consequently, it is demonstrated a recognition rate with a success of 96.76%"
    test_keywords = ["Convolutional neural network",
      "Deep learning",
      "Handwriting analysis",
      "Handwritten Arabic word recognition"]

    # Obtenir les recommandations
    recommendations = recommender.recommend(test_title, test_abstract, test_keywords)

    # Afficher les recommandations
    print("\nJournaux recommandés:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['journal_name']}")
        print(f"   Similarité: {rec['similarity']:.3f}")
        print(f"   Score d'impact: {rec['journal_score']:.3f}\n")
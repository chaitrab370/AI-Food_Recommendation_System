# === File: recommender/text_recommender.py ===

import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import clean_text

DATA_PATH = "model/recipe_data.csv"
EMBEDDINGS_PATH = "model/recipe_embeddings.pkl"

class RecipeRecommender:
    def __init__(self):  # ✅ Corrected constructor
        self.df = pd.read_csv(DATA_PATH)
        self.df.fillna("", inplace=True)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self._load_or_generate_embeddings()

    def _preprocess_row(self, row):
        return (
            clean_text(row['title']) + " " +
            clean_text(row['ingredients']) + " " +
            clean_text(row['description']) + " " +
            clean_text(row['instructions'])
        )

    def _load_or_generate_embeddings(self):
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            texts = self.df.apply(self._preprocess_row, axis=1).tolist()
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(self.embeddings, f)

    def recommend(self, query, food_type_filter=None, top_n=5):
        query_embedding = self.model.encode([clean_text(query)])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_idx = scores.argsort()[-top_n:][::-1]
        results = self.df.iloc[top_idx]

        # ✅ Filter recipes if a specific food type is selected
        if food_type_filter and food_type_filter != "Any":
            results = results[
                results["food_types"].str.lower().str.contains(food_type_filter.lower())
            ]

        return results[[
            'title', 'ingredients', 'instructions', 'food_types',
            'description', 'average_rating', 'minutes'
        ]].reset_index(drop=True)

if __name__ == "__main__":  # ✅ Corrected main entry
    recommender = RecipeRecommender()
    print("✅ Embeddings generated and saved!")



    



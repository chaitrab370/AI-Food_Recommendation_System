# === File: recommender/image_classifier.py ===

import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from utils.preprocess import clean_text


class FoodImageClassifier:
    def __init__(self):
        # Load pre-trained ResNet50 model for food image recognition
        self.model = ResNet50(weights='imagenet')

    def predict_label(self, img_path):
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get predictions
        preds = self.model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]

        # Format: [('pizza', 0.94), ('cheeseburger', 0.02), ('ice_cream', 0.01)]
        return [(label.lower().replace("_", " "), round(prob, 2)) for (_, label, prob) in decoded]


class ImageBasedRecommender:
    def __init__(self):
        # Load cleaned dataset with food_types
        self.df = pd.read_csv("model/recipe_data.csv")
        self.df.fillna("", inplace=True)
        self.classifier = FoodImageClassifier()

    def recommend_from_image(self, img_path, food_type_filter=None, top_n=5):
        # Step 1: Predict top food label from image
        top_class = self.classifier.predict_label(img_path)[0][0]

        # Step 2: Find recipes that match the predicted label
        matches = self.df[
            self.df['title'].str.contains(top_class, case=False) |
            self.df['ingredients'].str.contains(top_class, case=False) |
            self.df['description'].str.contains(top_class, case=False)
        ]

        # Step 3: Filter based on selected food_types (if not 'Any')
        if food_type_filter and food_type_filter.lower() != "any":
            matches = matches[matches['food_types'].str.lower().str.contains(food_type_filter.lower())]

        return matches.head(top_n).reset_index(drop=True)
if __name__ == "__main__":
    classifier = FoodImageClassifier()
    test_image = "E:/pictures-of-pizza-23-1.jpg"  # Replace with a valid path to a food image
    predictions = classifier.predict_label(test_image)
    print("🍽️ Top Predictions from Image:")
    for label, prob in predictions:
        print(f" - {label}: {prob}")


# AI-Food_Recommendation_System

## Overview

The AI-Powered Food Recommendation System helps users discover suitable recipes using text inputs (like ingredients or dish names) or food images. It simplifies meal decisions by offering personalized, intelligent recipe suggestions using Natural Language Processing (NLP) and Computer Vision techniques.

## Objectives
Recommend recipes from text or image input.
Provide personalized suggestions based on user preferences.
Integrate NLP and image models for accurate recipe matching.
Offer a clean, interactive interface for easy food discovery.

## Technologies Used
Programming Language: Python
Deep Learning: TensorFlow, PyTorch
NLP Model: Sentence-BERT (all-MiniLM-L6-v2)
Image Model: ResNet50
Similarity Measure: Cosine Similarity
Web Interface: Flask / Gradio
Dataset: recipeData.csv (69,306 recipes from Kaggle)

## Data Preprocessing
Removed null and incomplete entries.
Cleaned and normalized text fields.
Converted list strings into readable format using ast.literal_eval().
Engineered new features like number of ingredients & steps.
Filtered top 1500 high-quality recipes for faster performance.

## Core Modules
Text Recommendation: Matches user text queries with recipes using Sentence-BERT embeddings.
Image Classification: Uses ResNet50 to predict food type from uploaded images.
Chatbot Assistant: Guides users via simple rule-based replies.
Favorites Management: Lets users save and view favorite recipes.

# === File: utils/preprocess.py ===

import pandas as pd
import os
import ast

def clean_text(text):
    """Lowercase and strip spaces."""
    if isinstance(text, str):
        return text.strip().lower()
    return ""

def clean_and_save_recipe_data(input_path="recipeData.csv", output_path="model/recipe_data.csv", limit=1500):
    df = pd.read_csv(input_path)

    # Basic cleaning
    df = df[df['ingredients'].notnull() & df['steps'].notnull() & df['name'].notnull()]
    df['title'] = df['name']

    # Convert stringified lists to proper text
    df['ingredients'] = df['ingredients'].apply(lambda x: ", ".join(ast.literal_eval(x)) if pd.notnull(x) else "")
    df['instructions'] = df['steps'].apply(lambda x: ". ".join(ast.literal_eval(x)) if pd.notnull(x) else "")

    # Safely parse food_types (handle strings and malformed lists)
    def safe_parse_food_types(x):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return ", ".join(parsed)
            return str(parsed)
        except:
            return str(x)

    df['food_types'] = df['food types'].apply(safe_parse_food_types)

    # Select required columns
    final_df = df[[
        'title', 'ingredients', 'instructions', 'food_types', 'description',
        'average_rating', 'minutes', 'n_ingredients', 'n_steps', 'score'
    ]].dropna().head(limit)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    clean_and_save_recipe_data()


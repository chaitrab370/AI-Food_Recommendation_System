# === File: app.py ===

import gradio as gr
import pandas as pd
import tempfile
from recommender.text_recommender import RecipeRecommender
from recommender.image_classifier import ImageBasedRecommender

# Load models
text_model = RecipeRecommender()
image_model = ImageBasedRecommender()
saved_favorites = []
chat_history = []

# Dynamically extract food_types for dropdown
food_type_options = ["Any"] + sorted(text_model.df['food_types'].dropna().unique().tolist())

def format_results(results_df):
    if results_df.empty:
        return "❌ No matching recipes found."
    
    formatted = ""
    for _, row in results_df.iterrows():
        formatted += f"\n\n🍽 **{row['title']}**\n"
        formatted += f"🡢 *Ingredients:* {row['ingredients']}\n"
        formatted += f"📋 *Instructions:* {row['instructions']}\n"
        formatted += f"🥗 *Food Type:* {row['food_types']}\n"
        formatted += f"⏱️ *Time:* {row['minutes']} minutes\n"
        formatted += f"⭐ *Rating:* {row['average_rating']}\n"
    return formatted

def recommend_from_text(query, food_type):
    return format_results(text_model.recommend(query, food_type_filter=food_type))

def recommend_from_image(img, food_type):
    if img is None:
        return "⚠ Please upload an image."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        results = image_model.recommend_from_image(tmp.name, food_type_filter=food_type)
        return format_results(results)


def save_favorite(title):
    title = title.strip().lower()

    # Try exact (case-insensitive) match
    matched = text_model.df[text_model.df["title"].str.lower().str.strip() == title]

    # If no exact match, try partial match
    if matched.empty:
        matched = text_model.df[text_model.df["title"].str.lower().str.contains(title)]

    if not matched.empty:
        saved_favorites.append(matched.iloc[0].to_dict())
        return f"✅ Saved: {matched.iloc[0]['title']}"
    else:
        return "❌ Recipe not found. Try typing a partial name."

def view_favorites():
    if not saved_favorites:
        return "📬 No favorites yet."
    return format_results(pd.DataFrame(saved_favorites))

def chatbot_response(user_input):
    """Rule-based chatbot for food-related queries."""
    user_input_clean = user_input.strip().lower()
    response = ""

    if any(word in user_input_clean for word in ["spicy", "hot"]):
        response = "🌶️ Try Spicy Fried Rice or Masala Dosa!"
    elif any(word in user_input_clean for word in ["sweet", "dessert"]):
        response = "🍰 You might like Gulab Jamun or Chocolate Mousse!"
    elif any(word in user_input_clean for word in ["quick", "fast", "easy"]):
        response = "⏱️ Try Grilled Cheese Sandwich or 15-Minute Tomato Pasta!"
    elif "healthy" in user_input_clean:
        response = "🥗 Quinoa Salad or Steamed Veggies are great healthy options!"
    else:
        suggestion = text_model.df.sample(1)["title"].values[0]
        response = f"🤔 Not sure, but you might like: **{suggestion}**"

    chat_history.append((user_input, response))
    return chat_history

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🍽️ AI Food Recommender System")
    gr.Markdown("Search by text, image, or ask the chatbot!")

    with gr.Tab("📝 Text Search"):
        with gr.Row():
            inp = gr.Textbox(label="Enter ingredients or recipe name")
            food_type = gr.Dropdown(food_type_options, value="Any", label="Food Type")
        out = gr.Markdown()
        btn = gr.Button("Recommend")
        btn.click(recommend_from_text, [inp, food_type], out)

    with gr.Tab("🖼️ Image Search"):
        img = gr.Image(type="pil")
        food_type2 = gr.Dropdown(food_type_options, value="Any", label="Food Type")
        out2 = gr.Markdown()
        btn2 = gr.Button("Predict & Recommend")
        btn2.click(recommend_from_image, [img, food_type2], out2)

    with gr.Tab("❤️ Favorites"):
        title = gr.Textbox(label="Recipe title to save (case-insensitive)")
        save_btn = gr.Button("Save Favorite")
        msg = gr.Textbox(label="Save Status")
        save_btn.click(save_favorite, title, msg)
        view_btn = gr.Button("View Favorites")
        fav_out = gr.Markdown()
        view_btn.click(view_favorites, outputs=fav_out)

    with gr.Tab("🤖 Chatbot"):
        chatbot_ui = gr.Chatbot()
        user_input = gr.Textbox(label="Ask something about food...")
        send_btn = gr.Button("Send")
        send_btn.click(chatbot_response, user_input, chatbot_ui)

# Fix: Correct the __name__ block
if __name__ == '__main__':
    demo.launch()


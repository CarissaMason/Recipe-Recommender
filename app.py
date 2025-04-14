import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re

# -----------------------------
# Cleaning Functions
# -----------------------------

# Fix capitalization like Ball'S → Ball's
def fix_title_case(text):
    if isinstance(text, str):
        fixed = text.title()
        return re.sub(r"'S\b", "'s", fixed)
    return text

# Clean ingredients or directions
def clean_list_column(text):
    try:
        items = ast.literal_eval(text)
        if isinstance(items, list):
            return ", ".join(item.strip().strip("\"'") for item in items)
    except:
        pass
    return str(text).strip("[]\"'")

# -----------------------------
# Load and Clean Data
# -----------------------------

df = pd.read_csv("cleaned.csv")
df = df[['title', 'ingredients', 'directions']].dropna()

# Apply cleaning
df['title'] = df['title'].apply(fix_title_case)
df['ingredients_clean'] = df['ingredients'].apply(clean_list_column)
df['directions_clean'] = df['directions'].apply(clean_list_column)

# Vegetarian flag
non_veg_keywords = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'bacon', 'turkey', 'lamb', 'ham', 'sausage']
df['vegetarian'] = df['ingredients_clean'].apply(lambda x: not any(meat in x for meat in non_veg_keywords))

# TF-IDF setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_clean'])

# -----------------------------
# Recommendation Function
# -----------------------------

def recommend_recipes(user_ingredients, top_n=5, vegetarian_only=False):
    user_vec = vectorizer.transform([user_ingredients.lower()])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
    top_indices = similarity_scores[0].argsort()[::-1]
    results = df.iloc[top_indices]
    if vegetarian_only:
        results = results[results['vegetarian']]
    return results[['title', 'ingredients_clean', 'directions_clean']].head(top_n)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Recipe Recommender")
st.markdown("""
Welcome! Just enter the ingredients you have, and I'll recommend recipes you can make.  
Use the checkbox to filter for vegetarian options. Click below each recipe to view full directions.
""")
st.markdown("---")

user_input = st.text_input("Enter ingredients (comma separated):", "sugar, eggs, rice")
vegetarian = st.checkbox("Vegetarian only")
top_n = st.slider("How many recipes would you like to see?", min_value=1, max_value=10, value=5)

if st.button("Find Recipes"):
    recs = recommend_recipes(user_input, top_n=top_n, vegetarian_only=vegetarian)
    for i, row in recs.iterrows():
        st.markdown(f"### 🍽️ {row['title']}")
        st.markdown(f"**🧂 Ingredients:** {row['ingredients_clean']}")

        with st.expander("📖 View full directions"):
            try:
                steps = ast.literal_eval(row['directions_clean'])
                if isinstance(steps, list):
                    for step in steps:
                        for sentence in step.split("."):
                            sentence = sentence.strip()
                            if sentence:
                                st.markdown(f"- {sentence}.")
                else:
                    raise ValueError("Not a list")
            except:
                for sentence in row['directions_clean'].split("."):
                    sentence = sentence.strip().lstrip(",;:- ")
                    if sentence:
                        st.markdown(f"- {sentence}.")
        
        st.markdown("---")

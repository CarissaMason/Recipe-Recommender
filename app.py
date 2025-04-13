import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("cleaned.csv")
df = df[['title', 'ingredients', 'directions']].dropna()
df['ingredients_clean'] = df['ingredients'].str.lower().str.replace(r"[\[\]'\"{}]", '', regex=True)

# Vegetarian flag
non_veg_keywords = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'bacon', 'turkey', 'lamb', 'ham', 'sausage']
df['vegetarian'] = df['ingredients_clean'].apply(lambda x: not any(meat in x for meat in non_veg_keywords))

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_clean'])

# Recommender
def recommend(user_ingredients, top_n=5, vegetarian_only=False):
    user_vec = vectorizer.transform([user_ingredients.lower()])
    sim = cosine_similarity(user_vec, tfidf_matrix)
    idx = sim[0].argsort()[::-1]
    results = df.iloc[idx]
    if vegetarian_only:
        results = results[results['vegetarian']]
    return results[['title', 'ingredients_clean', 'directions']].head(top_n)

# Streamlit UI
st.title("Ingredient-Based Recipe Recommender")

user_input = st.text_input("Enter ingredients (comma separated):", "chicken, garlic, rice")
vegetarian = st.checkbox("Vegetarian only")

if st.button("Find Recipes"):
    recs = recommend(user_input, vegetarian_only=vegetarian)
    for i, row in recs.iterrows():
        st.subheader(row['title'])
        st.markdown(f"**Ingredients:** {row['ingredients_clean']}")
        st.markdown(f"**Directions:** {row['directions'][:300]}...")

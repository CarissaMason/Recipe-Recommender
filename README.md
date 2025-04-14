# Recipe Recommender

A user-friendly web app that recommends recipes based on the ingredients you have! Built using **Python** & **Streamlit**, this app lets you:

- Find recipes from a cleaned dataset  
- Filter for vegetarian-only meals  
- View full step-by-step directions  
- Get clean, readable outputs

---

## App

Check out the app here [Recipe Recommender App](https://recipe-recommender-r9sx4zzpppqkg8gzdah2pv.streamlit.app/)

---

## Features

- **Ingredient-based search**  
  Type a few ingredients (e.g., `chicken, rice, garlic`) and get recipes that match.

- **Vegetarian filter**  
  Check a box to only show vegetarian recipes.

- **Clean, bulleted instructions**  
  Each recipe shows clearly formatted steps with punctuation cleaned and stripped.

- **Fast and interactive**  
  Built with Streamlit and scikit-learn's `TfidfVectorizer` for quick similarity comparisons.

---

## Dataset

This project uses the [RecipeNLG Dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg), which includes:

- 2M+ recipes  
- Title, ingredients, and direction fields  
- Some messy formatting — which I cleaned before use!

---

## Data Cleaning Highlights

- Fixed inconsistent capitalization (e.g., `Peanut Butter Ball'S` → `Peanut Butter Ball's`)
- Removed brackets, quotes, commas from directions
- Stripped list-based text into clean strings
- Handled both raw strings and Python-literal lists
- Added vegetarian detection using keyword filtering

---

## How It Works

1. **Ingredients are vectorized** using TF-IDF  
2. **Cosine similarity** is calculated between user input and recipe ingredient lists  
3. Top matching recipes are displayed in an organized format

---

## Installation & Running

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt

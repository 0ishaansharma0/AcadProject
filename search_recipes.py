# Improved, fixed, and commented semantic + fuzzy recipe search
# Save this as search_recipes.py and run. Adjust DATA_PATH and model_name as needed.

import os
from pathlib import Path
import pickle
import re
import unicodedata
import numpy as np
import pandas as pd

# embeddings & fuzzy
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# optional torch only used for device detection and saving/loading tensors
import torch

# ---------- config ----------
DATA_PATH = Path("data/recipes.xlsx")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Choose model: strong but slow -> "all-mpnet-base-v2"; fast & decent -> "all-MiniLM-L6-v2"
MODEL_NAME = "all-MiniLM-L6-v2"   # default: faster and smaller for development
BATCH_SIZE = 64                   # pass to model.encode to limit memory/throughput spikes
TOP_PREFILTER = 200               # only fuzzy-match top semantic candidates
MIN_SCORE = 0.30                  # default min combined score threshold
SEM_FUZZY_WEIGHT = (0.7, 0.3)     # (semantic_weight, fuzzy_weight)

# ---------- helpers ----------
def normalize_text(s, keep_digits=False):
    """Lowercase, strip, normalize unicode, remove punctuation.
       Set keep_digits=True if numerical tokens (e.g., '2 eggs') matter."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    if keep_digits:
        s = re.sub(r"[^0-9a-z\s]", " ", s)
    else:
        s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_col(df, options):
    for name in options:
        if name in df.columns:
            return name
    return None

# ---------- load data ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing {DATA_PATH}. Put your recipes.xlsx in data/ or change DATA_PATH.")

df = pd.read_excel(DATA_PATH)

# Robust column detection
name_col = find_col(df, ["recipe_name_orig", "recipe_name", "_recipe_name_orig", "RecipeName", "translatedRecipeName", "recipe_name_org"])
ing_col  = find_col(df, ["ingredient_name_org", "ingredient_name", "ingredient", "Ingredient_Name", "food_name_org", "food_name"])
food_col = find_col(df, ["food_name_org", "food_name", "food", "Food_Name", "translatedIngredients"])

if not name_col:
    raise KeyError("No recipe name column found. Check your Excel headers (tried several options).")

# Combine ingredient columns (graceful to missing columns)
def combine_row_text(row):
    parts = []
    if ing_col and ing_col in row and pd.notna(row[ing_col]):
        parts.append(str(row[ing_col]))
    if food_col and food_col in row and pd.notna(row[food_col]):
        parts.append(str(row[food_col]))
    return " ".join(parts).strip()

df = df.dropna(subset=[name_col]).reset_index(drop=True)
df["combined_text"] = df.apply(combine_row_text, axis=1)
print(f"Loaded {len(df)} rows. Using recipe-name column: '{name_col}'. ingredient/food columns: '{ing_col}', '{food_col}'")

# Group into one document per recipe (join ingredient rows)
grouped = (
    df.groupby(name_col)["combined_text"]
      .apply(lambda texts: " ".join([t for t in texts.astype(str) if t.strip() != ""]))
      .reset_index(name="full_text")
)

# Keep original name column (string), normalized versions for encoding
grouped["recipe_clean"] = grouped[name_col].apply(lambda x: normalize_text(x if pd.notna(x) else ""))
grouped["clean_text"] = grouped["full_text"].apply(lambda x: normalize_text(x if pd.notna(x) else "", keep_digits=True))

print(f"Number of unique recipes after grouping: {len(grouped)}")
if grouped[name_col].duplicated().any():
    print("Warning: duplicate recipe names found after grouping. Consider using a unique id to distinguish.")

# ---------- load model & compute embeddings ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}. Model: {MODEL_NAME} (batch_size={BATCH_SIZE})")
model = SentenceTransformer(MODEL_NAME, device=device)

# Compose text for embedding: name + ingredients (you can experiment with order/sep)
texts_to_encode = (grouped["recipe_clean"] + " " + grouped["clean_text"]).tolist()

# encode with batching to avoid OOM
recipe_embeddings = model.encode(
    texts_to_encode,
    batch_size=BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=device
)
# ensure float32
if recipe_embeddings.dtype != torch.float32 and hasattr(recipe_embeddings, "to"):
    recipe_embeddings = recipe_embeddings.to(torch.float32)

# Save artifacts for reuse
# - embeddings as numpy for portability
np.save(MODELS_DIR / "recipe_embeddings.npy", recipe_embeddings.cpu().numpy())
grouped.to_pickle(MODELS_DIR / "recipes_mapping.pkl")

print("Saved embeddings and mapping to models/")

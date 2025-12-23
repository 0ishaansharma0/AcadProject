import streamlit as st
from search_recipes import search_recipe_semantic

st.set_page_config(page_title="Recipe Search", layout="centered")

st.title("🍽️ Recipe Search")
st.write("Semantic + fuzzy recipe search")

query = st.text_input("Search for a recipe", placeholder="e.g. phulka, chicken curry")

top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

boost_keywords = st.text_input(
    "Optional boost keywords (comma separated)",
    placeholder="e.g. indian, rice, curry"
)

if st.button("Search"):
    if not query.strip():
        st.warning("Enter a search query")
    else:
        boosts = [b.strip().lower() for b in boost_keywords.split(",") if b.strip()]
        results = search_recipe_semantic(
            query,
            top_k=top_k,
            boost_keywords=boosts if boosts else None
        )

        if not results:
            st.error("No results found")
        else:
            for r in results:
                st.subheader(r["recipe_name"])
                st.caption(f"Score: {r['score']}")
                st.write(r["ingredients_preview"])
                if r.get("fallback"):
                    st.warning("Fallback result")
                st.divider()

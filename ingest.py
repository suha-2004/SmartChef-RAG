import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load recipes from API
with open("recipes_api.json", "r", encoding="utf-8") as file:
    recipes = json.load(file)

print(f"Loaded {len(recipes)} recipes.")


documents = []

for recipe in recipes:

    name = recipe.get("strMeal", "Unknown Recipe")
    category = recipe.get("strCategory", "Unknown")
    area = recipe.get("strArea", "Unknown")

    ingredients = []

    # TheMealDB stores ingredients as strIngredient1 ... strIngredient20
    for i in range(1, 21):

        ingredient = recipe.get(f"strIngredient{i}")
        measure = recipe.get(f"strMeasure{i}")

        if ingredient and ingredient.strip():

            ingredient = ingredient.strip()

            if measure and measure.strip():
                ingredients.append(f"{measure.strip()} {ingredient}")
            else:
                ingredients.append(ingredient)

    instructions = recipe.get("strInstructions", "")

    recipe_text = f"""
RECIPE NAME: {name}

CATEGORY: {category}

CUISINE: {area}

INGREDIENTS:
{chr(10).join(ingredients)}

INSTRUCTIONS:
{instructions}

SOURCE: TheMealDB
"""

    documents.append(
        Document(
            page_content=recipe_text,
            metadata={
                "recipe_name": name,
                "category": category,
                "cuisine": area,
                "source": "TheMealDB"
            }
        )
    )


print(f"Created {len(documents)} recipe documents.")


# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# Create embeddings
print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS database
print("Creating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# Save vector database
vectorstore.save_local("vectorstore")

print()
print("====================================")
print("RAG DATABASE CREATED SUCCESSFULLY!")
print("====================================")
print(f"Recipes: {len(documents)}")
print(f"Chunks: {len(chunks)}")
print("Location: vectorstore/")
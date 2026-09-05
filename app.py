import streamlit as st
import os
import re
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SmartChef AI",
    page_icon="👨‍🍳",
    layout="wide"
)


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing from the .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.recipe-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("👨‍🍳 SmartChef AI")

    st.write(
        "Personal Recipe Intelligence powered by "
        "RAG + Gemini."
    )

    st.divider()

    st.subheader("✨ Features")

    st.write("🔎 Recipe search")
    st.write("🥕 Ingredient matching")
    st.write("⏱️ Cooking-time filtering")
    st.write("🛒 Shopping list")
    st.write("🔄 Ingredient substitutions")
    st.write("📚 RAG-based answers")
    st.write("🌍 Multiple cuisines")

    st.divider()

    st.subheader("🛠️ Technologies")

    st.write("Python")
    st.write("Streamlit")
    st.write("LangChain")
    st.write("FAISS")
    st.write("Hugging Face")
    st.write("Gemini")
    st.write("TheMealDB API")


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">👨‍🍳 SmartChef AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find recipes using your ingredients with RAG-powered AI.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INGREDIENT LIST
# =========================================================

INGREDIENTS = [

    # Meat / Seafood
    "chicken",
    "mutton",
    "fish",
    "prawns",
    "beef",
    "lamb",

    # Grains / Main ingredients
    "rice",
    "basmati rice",
    "pasta",
    "noodles",
    "bread",
    "flour",
    "semolina",
    "lentils",
    "chickpeas",
    "beans",

    # Vegetables
    "potato",
    "sweet potato",
    "cauliflower",
    "carrot",
    "capsicum",
    "green peas",
    "peas",
    "cabbage",
    "mushroom",
    "broccoli",
    "spinach",
    "corn",
    "tomato",
    "onion",
    "garlic",
    "ginger",
    "green chilli",
    "chilli",

    # Dairy / Eggs
    "paneer",
    "cheese",
    "yogurt",
    "curd",
    "milk",
    "cream",
    "butter",
    "eggs",
    "egg",

    # Fruits
    "banana",
    "apple",
    "orange",
    "grapes",

    # Basic ingredients
    "salt",
    "pepper",
    "oil",
    "sugar",
    "chocolate",
    "water",
    "soy sauce"
]


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# =========================================================
# LOAD FAISS
# =========================================================

@st.cache_resource
def load_vectorstore():

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()


# =========================================================
# LOAD GEMINI
# =========================================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2
    )


llm = load_llm()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_ingredients(text):
    """
    Detect ingredients mentioned by the user.
    """

    text = text.lower()

    sorted_ingredients = sorted(
        INGREDIENTS,
        key=len,
        reverse=True
    )

    found = []

    for ingredient in sorted_ingredients:

        pattern = r"\b" + re.escape(ingredient) + r"\b"

        if re.search(pattern, text):

            if ingredient not in found:
                found.append(ingredient)

    return found


def extract_recipe_ingredients(recipe_text):
    """
    Extract ingredients from the actual INGREDIENTS
    section of a retrieved recipe.
    """

    recipe_text = recipe_text.lower()

    if "ingredients:" not in recipe_text:
        return []

    ingredients_section = recipe_text.split(
        "ingredients:",
        1
    )[1]

    if "instructions:" in ingredients_section:

        ingredients_section = ingredients_section.split(
            "instructions:",
            1
        )[0]

    ingredients = []

    for line in ingredients_section.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove bullet points if present
        line = re.sub(
            r"^[•\-\*\d\.\)\s]+",
            "",
            line
        )

        if line:
            ingredients.append(line)

    return ingredients


def extract_time(text):
    """
    Extract cooking time in minutes.
    """

    text = text.lower()

    patterns = [
        r"(\d+)\s*minutes?",
        r"(\d+)\s*mins?",
        r"(\d+)\s*hours?"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            value = int(match.group(1))

            if "hour" in pattern:
                value *= 60

            return value

    return None


def get_recipe_time(recipe_text):

    return extract_time(recipe_text)


def find_missing_ingredients(
    recipe_ingredients,
    user_ingredients
):
    """
    Find recipe ingredients that the user does not have.
    """

    user_ingredients = [
        x.lower().strip()
        for x in user_ingredients
    ]

    missing = []

    for recipe_ingredient in recipe_ingredients:

        recipe_ingredient = recipe_ingredient.lower().strip()

        found = False

        for user_item in user_ingredients:

            user_item = user_item.lower().strip()

            if (
                recipe_ingredient in user_item
                or user_item in recipe_ingredient
            ):
                found = True
                break

        if not found:

            missing.append(recipe_ingredient)

    return missing


def find_substitution(ingredient):

    substitutions = {

        "yogurt": [
            "Buttermilk",
            "Coconut milk",
            "Cream"
        ],

        "curd": [
            "Yogurt",
            "Buttermilk"
        ],

        "egg": [
            "Tofu",
            "Mashed potato",
            "Flaxseed mixture"
        ],

        "eggs": [
            "Tofu",
            "Mashed potato",
            "Flaxseed mixture"
        ],

        "butter": [
            "Oil",
            "Ghee",
            "Margarine"
        ],

        "milk": [
            "Coconut milk",
            "Soy milk",
            "Oat milk"
        ],

        "cheese": [
            "Paneer",
            "Tofu",
            "Nutritional yeast"
        ],

        "cream": [
            "Coconut milk",
            "Yogurt",
            "Milk"
        ],

        "potato": [
            "Sweet potato",
            "Cauliflower"
        ],

        "rice": [
            "Quinoa",
            "Couscous"
        ],

        "oil": [
            "Butter",
            "Ghee"
        ],

        "tomato": [
            "Tomato puree",
            "Tamarind",
            "Red bell pepper"
        ],

        "chicken": [
            "Paneer",
            "Tofu",
            "Mushroom"
        ],

        "fish": [
            "Chicken",
            "Paneer",
            "Tofu"
        ]

    }

    return substitutions.get(
        ingredient.lower(),
        []
    )


# =========================================================
# USER INPUT
# =========================================================

st.subheader("🥕 What ingredients do you have?")

selected_ingredients = st.multiselect(
    "Select your available ingredients:",
    INGREDIENTS
)


custom_ingredients = st.text_input(
    "Or type additional ingredients:",
    placeholder="Example: garlic, onion, chicken"
)


# =========================================================
# COOKING TIME
# =========================================================

st.subheader("⏱️ Cooking Time")

max_time = st.slider(
    "Maximum cooking time (minutes)",
    min_value=10,
    max_value=180,
    value=60,
    step=10
)


# =========================================================
# USER QUERY
# =========================================================

user_query = st.text_area(
    "🍽️ What do you want to cook?",
    placeholder=(
        "Example: I have chicken, rice and onion. "
        "Suggest something quick."
    ),
    height=100
)


search_button = st.button(
    "🔎 Find Recipes",
    use_container_width=True
)


# =========================================================
# SEARCH
# =========================================================

if search_button:

    if (
        not selected_ingredients
        and not custom_ingredients
        and not user_query
    ):

        st.warning(
            "Please select ingredients or enter a recipe request."
        )

        st.stop()


    # =====================================================
    # COLLECT USER INGREDIENTS
    # =====================================================

    user_ingredients = selected_ingredients.copy()

    if custom_ingredients:

        custom_found = extract_ingredients(
            custom_ingredients
        )

        for ingredient in custom_found:

            if ingredient not in user_ingredients:

                user_ingredients.append(
                    ingredient
                )


    # Also detect ingredients from the complete user query
    if user_query:

        query_ingredients = extract_ingredients(
            user_query
        )

        for ingredient in query_ingredients:

            if ingredient not in user_ingredients:

                user_ingredients.append(
                    ingredient
                )


    # =====================================================
    # SEARCH QUERY
    # =====================================================

    query_parts = []

    if user_query:

        query_parts.append(
            user_query
        )

    if user_ingredients:

        query_parts.append(
            "Ingredients: "
            + ", ".join(user_ingredients)
        )

    query = " ".join(query_parts)


    # =====================================================
    # RETRIEVE DOCUMENTS
    # =====================================================

    docs = vectorstore.similarity_search(
        query,
        k=10
    )


    if not docs:

        st.warning(
            "No recipes were found in the knowledge base."
        )

        st.stop()


    # =====================================================
    # RANK RESULTS
    # =====================================================

    ranked_docs = []

    for document in docs:

        text = document.page_content.lower()

        score = 0


        # -------------------------------------------------
        # Ingredient matching
        # -------------------------------------------------

        for ingredient in user_ingredients:

            if ingredient.lower() in text:

                score += 3


        # -------------------------------------------------
        # Cooking time
        # -------------------------------------------------

        recipe_time = get_recipe_time(
            document.page_content
        )

        if recipe_time:

            if recipe_time <= max_time:

                score += 8

            else:

                score -= 8


        ranked_docs.append(
            (score, document)
        )


    ranked_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )


    # Remove score
    ranked_docs = [
        document
        for score, document in ranked_docs
    ]


    # =====================================================
    # DISPLAY TOP MATCHES
    # =====================================================

    st.subheader(
        "🍽️ Recommended Recipes"
    )


    for document in ranked_docs[:5]:

        recipe_text = document.page_content

        metadata = document.metadata

        recipe_name = metadata.get(
            "recipe_name",
            "Recipe"
        )

        category = metadata.get(
            "category",
            "Unknown"
        )

        cuisine = metadata.get(
            "cuisine",
            "Unknown"
        )

        recipe_time = get_recipe_time(
            recipe_text
        )


        with st.expander(
            f"🍴 {recipe_name}"
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write(
                    f"**📂 Category:** {category}"
                )

            with col2:

                st.write(
                    f"**🌍 Cuisine:** {cuisine}"
                )

            with col3:

                if recipe_time:

                    st.write(
                        f"**⏱️ Time:** "
                        f"{recipe_time} minutes"
                    )

                else:

                    st.write(
                        "**⏱️ Time:** Not available"
                    )

            st.divider()

            st.write(
                recipe_text
            )


    # =====================================================
    # INGREDIENT ANALYSIS
    # =====================================================

    missing = []

    if user_ingredients:

        st.divider()

        st.subheader(
            "🥕 Ingredient Analysis"
        )


        best_recipe = ranked_docs[0]


        recipe_ingredients = extract_recipe_ingredients(
            best_recipe.page_content
        )


        missing = find_missing_ingredients(
            recipe_ingredients,
            user_ingredients
        )


        if missing:

            st.warning(
                "Missing ingredients: "
                + ", ".join(missing)
            )

        else:

            st.success(
                "🎉 You have all detected ingredients "
                "needed for this recipe!"
            )


    # =====================================================
    # SHOPPING LIST
    # =====================================================

    if missing:

        st.divider()

        st.subheader(
            "🛒 Shopping List"
        )


        shopping_list = "\n".join(
            f"- {item}"
            for item in missing
        )


        st.text_area(
            "Ingredients to buy:",
            shopping_list,
            height=150
        )


        st.download_button(
            "⬇️ Download Shopping List",
            shopping_list,
            file_name="smartchef_shopping_list.txt"
        )


    # =====================================================
    # SUBSTITUTIONS
    # =====================================================

    if missing:

        st.divider()

        st.subheader(
            "🔄 Ingredient Substitutions"
        )


        for ingredient in missing:

            alternatives = find_substitution(
                ingredient
            )


            if alternatives:

                st.write(
                    f"**{ingredient.title()} →** "
                    + ", ".join(alternatives)
                )


    # =====================================================
    # RAG + GEMINI
    # =====================================================

    st.divider()

    st.subheader(
        "🤖 SmartChef AI Recommendation"
    )


    context = "\n\n".join(
        document.page_content
        for document in ranked_docs[:5]
    )


    prompt = f"""
You are SmartChef AI, a recipe recommendation assistant.

The user asked:

{query}

The user's available ingredients are:

{", ".join(user_ingredients)}

Maximum cooking time:

{max_time} minutes


Use ONLY the recipes retrieved from the SmartChef
knowledge base below.

Do NOT invent recipes that are not present in the
retrieved knowledge base.

Retrieved recipes:

{context}


Give the user:

1. The best matching recipe.
2. Why it matches their ingredients.
3. Cooking time.
4. Missing ingredients, if any.
5. Possible substitutions when appropriate.
6. A short step-by-step preparation method.

If none of the retrieved recipes is suitable,
honestly say that no suitable recipe was found.

Keep the answer clear and beginner-friendly.
"""


    try:

        response = llm.invoke(
            prompt
        )

        st.write(
            response.content
        )

    except Exception as e:

        st.error(
            f"Gemini error: {e}"
        )


    # =====================================================
    # SOURCES
    # =====================================================

    st.divider()

    st.subheader(
        "📚 Sources Used"
    )

    st.write(
        "These recipes were retrieved from the "
        "SmartChef RAG knowledge base."
    )


    for document in ranked_docs[:5]:

        metadata = document.metadata

        recipe_name = metadata.get(
            "recipe_name",
            "Recipe"
        )

        source_name = metadata.get(
            "source",
            "TheMealDB"
        )


        with st.expander(
            f"🍽️ {recipe_name}"
        ):

            col1, col2, col3 = st.columns(3)


            with col1:

                st.write(
                    f"**📄 Source:** {source_name}"
                )


            with col2:

                st.write(
                    f"**📂 Category:** "
                    f"{metadata.get('category', 'Unknown')}"
                )


            with col3:

                st.write(
                    f"**🌍 Cuisine:** "
                    f"{metadata.get('cuisine', 'Unknown')}"
                )


            st.divider()


            st.write(
                document.page_content
            )


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.subheader(
    "🧠 How SmartChef Works"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.write("### 1️⃣ API")

    st.write(
        "Recipes are collected from TheMealDB."
    )


with col2:

    st.write("### 2️⃣ Embeddings")

    st.write(
        "Recipes are converted into numerical vectors."
    )


with col3:

    st.write("### 3️⃣ RAG")

    st.write(
        "FAISS retrieves the most relevant recipes."
    )


with col4:

    st.write("### 4️⃣ Gemini")

    st.write(
        "Gemini generates the final recommendation."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "SmartChef AI • Recipe Intelligence using RAG + LLM"
)
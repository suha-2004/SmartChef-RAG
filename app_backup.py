import os
import re
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="SmartChef AI",
    page_icon="🍳",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:

    st.error(
        "Google API key not found. "
        "Please add GOOGLE_API_KEY to your .env file."
    )

    st.stop()


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .recipe-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    .small-text {
        font-size: 14px;
    }

    .feature-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🍳 SmartChef AI")

    st.write(
        "Your personal AI-powered recipe assistant."
    )

    st.divider()

    st.subheader("✨ Features")

    st.write("🥕 Ingredient-based recipes")
    st.write("⏱️ Cooking-time filtering")
    st.write("🛒 Missing ingredient detection")
    st.write("🔄 Ingredient substitutions")
    st.write("📋 Shopping list")
    st.write("📤 Upload your own recipes")
    st.write("📚 RAG source retrieval")

    st.divider()

    st.subheader("🧠 Technologies")

    st.write("🐍 Python")
    st.write("🦜 LangChain")
    st.write("🔎 FAISS")
    st.write("🤗 HuggingFace")
    st.write("✨ Gemini")
    st.write("🎈 Streamlit")

    st.divider()

    st.caption(
        "SmartChef AI uses Retrieval "
        "Augmented Generation (RAG)."
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🍳 SmartChef AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Personal Recipe Intelligence Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Find recipes using the ingredients you already have, "
    "discover substitutions, create shopping lists, "
    "and ask questions about your recipes."
)


# ============================================================
# INGREDIENT LIST
# ============================================================

INGREDIENTS = [

    # --------------------------------------------------------
    # MEAT / SEAFOOD
    # --------------------------------------------------------

    "chicken",
    "mutton",
    "fish",
    "prawns",
    "beef",
    "lamb",

    # --------------------------------------------------------
    # GRAINS / MAIN INGREDIENTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VEGETABLES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DAIRY / EGGS
    # --------------------------------------------------------

    "paneer",
    "cheese",
    "yogurt",
    "curd",
    "milk",
    "cream",
    "butter",
    "eggs",
    "egg",

    # --------------------------------------------------------
    # FRUITS
    # --------------------------------------------------------

    "banana",
    "apple",
    "orange",
    "grapes",

    # --------------------------------------------------------
    # BASIC INGREDIENTS
    # --------------------------------------------------------

    "salt",
    "pepper",
    "oil",
    "sugar",
    "chocolate",
    "water",
    "soy sauce"
]


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()


# ============================================================
# LOAD GEMINI
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )


llm = load_llm()


# ============================================================
# FUNCTIONS
# ============================================================

def extract_ingredients(text):

    text = text.lower()

    found = []

    for ingredient in INGREDIENTS:

        if ingredient in text:

            found.append(ingredient)

    return found


# ============================================================
# EXTRACT TIME
# ============================================================

def extract_time(text):

    text = text.lower()

    patterns = [

        r"(\d+)\s*minutes?",

        r"(\d+)\s*mins?",

        r"under\s+(\d+)",

        r"within\s+(\d+)"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            return int(
                match.group(1)
            )

    return None


# ============================================================
# GET RECIPE TIME
# ============================================================

def get_recipe_time(recipe_text):

    match = re.search(

        r"COOKING TIME:\s*(\d+)\s*minutes?",

        recipe_text,

        re.IGNORECASE
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# GET RECIPE INGREDIENTS
# ============================================================

def get_recipe_ingredients(recipe_text):

    recipe_text = recipe_text.lower()

    ingredients = []

    match = re.search(

        r"ingredients:(.*?)(?:cooking time:|instructions:)",

        recipe_text,

        re.IGNORECASE | re.DOTALL
    )

    if match:

        ingredient_section = match.group(1)

        for ingredient in INGREDIENTS:

            if ingredient in ingredient_section:

                ingredients.append(
                    ingredient
                )

    return ingredients


# ============================================================
# FIND MISSING INGREDIENTS
# ============================================================

def find_missing_ingredients(
    user_ingredients,
    recipe_ingredients
):

    missing = []

    for ingredient in recipe_ingredients:

        if ingredient not in user_ingredients:

            missing.append(
                ingredient
            )

    return missing


# ============================================================
# SUBSTITUTIONS
# ============================================================

def find_substitution(ingredient):

    substitutions = {

        "yogurt": [
            "Buttermilk",
            "Coconut milk",
            "Cream"
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
        ]
    }

    return substitutions.get(
        ingredient.lower(),
        []
    )


# ============================================================
# UPLOAD RECIPE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📤 Upload Your Own Recipe'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a TXT or PDF recipe and add it "
    "to SmartChef's knowledge base."
)


uploaded_file = st.file_uploader(

    "Choose a recipe file",

    type=[
        "txt",
        "pdf"
    ]
)


if uploaded_file is not None:

    if st.button(
        "➕ Add Recipe to SmartChef",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # TXT
            # ------------------------------------------------

            if uploaded_file.name.lower().endswith(
                ".txt"
            ):

                text = uploaded_file.read().decode(
                    "utf-8"
                )


            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            elif uploaded_file.name.lower().endswith(
                ".pdf"
            ):

                pdf_bytes = uploaded_file.getvalue()

                pdf_reader = PdfReader(
                    BytesIO(pdf_bytes)
                )

                pages = []

                for page in pdf_reader.pages:

                    page_text = page.extract_text()

                    if page_text:

                        pages.append(
                            page_text
                        )

                text = "\n".join(
                    pages
                )


            else:

                st.error(
                    "Unsupported file type."
                )

                st.stop()


            # ------------------------------------------------
            # CHECK TEXT
            # ------------------------------------------------

            if not text.strip():

                st.error(
                    "Could not extract text from this file."
                )

                st.stop()


            # ------------------------------------------------
            # CREATE DOCUMENT
            # ------------------------------------------------

            document = Document(

                page_content=text,

                metadata={
                    "source": uploaded_file.name
                }
            )


            # ------------------------------------------------
            # SPLIT DOCUMENT
            # ------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(

                chunk_size=800,

                chunk_overlap=100
            )


            chunks = text_splitter.split_documents(
                [document]
            )


            # ------------------------------------------------
            # ADD TO FAISS
            # ------------------------------------------------

            vectorstore.add_documents(
                chunks
            )


            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            vectorstore.save_local(
                "vectorstore"
            )


            # ------------------------------------------------
            # CLEAR CACHE
            # ------------------------------------------------

            st.cache_resource.clear()


            st.success(
                f"✅ {uploaded_file.name} "
                "was successfully added!"
            )

            st.info(
                "You can now ask SmartChef questions "
                "about this recipe."
            )


        except Exception as e:

            st.error(
                f"Error processing file: {e}"
            )


# ============================================================
# QUESTION SECTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🥕 What would you like to cook?'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Tell SmartChef what ingredients you have "
    "or describe what you want to prepare."
)


question = st.text_area(

    "",

    placeholder=(
        "Example: I have chicken, rice and onion. "
        "What can I cook in 30 minutes?"
    ),

    height=130
)


# ============================================================
# FIND RECIPES
# ============================================================

if st.button(
    "🔎 Find Recipes",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter some ingredients "
            "or a question."
        )

        st.stop()


    # ========================================================
    # EXTRACT USER INFORMATION
    # ========================================================

    user_ingredients = extract_ingredients(
        question
    )

    requested_time = extract_time(
        question
    )


    substitution_question = any(

        word in question.lower()

        for word in [

            "substitute",
            "substitution",
            "instead of",
            "don't have",
            "do not have",
            "without"

        ]
    )


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    with st.spinner(
        "🔎 Searching your recipe collection..."
    ):

        retrieved_docs = vectorstore.similarity_search(

            question,

            k=10
        )


    # ========================================================
    # RANK RECIPES
    # ========================================================

    scored_docs = []


    for document in retrieved_docs:

        recipe_text = document.page_content.lower()

        score = 0


        # ----------------------------------------------------
        # INGREDIENT MATCHING
        # ----------------------------------------------------

        for ingredient in user_ingredients:

            if ingredient in recipe_text:

                score += 3


        # ----------------------------------------------------
        # VEGETARIAN FILTER
        # ----------------------------------------------------

        if (

            "vegetable" in question.lower()

            or "vegetarian" in question.lower()

            or "veg" in question.lower()

        ):

            if "category: vegetarian" in recipe_text:

                score += 5


            if "category: non-vegetarian" in recipe_text:

                score -= 10


        # ----------------------------------------------------
        # COOKING TIME
        # ----------------------------------------------------

        recipe_time = get_recipe_time(
            document.page_content
        )


        if requested_time is not None:

            if recipe_time is not None:

                if recipe_time <= requested_time:

                    score += 8

                else:

                    score -= 8


        scored_docs.append(

            (
                score,
                recipe_time,
                document
            )
        )


    # ========================================================
    # SORT
    # ========================================================

    scored_docs.sort(

        key=lambda x: x[0],

        reverse=True
    )


    # ========================================================
    # TIME FILTER
    # ========================================================

    if requested_time is not None:

        time_filtered_docs = []


        for (
            score,
            recipe_time,
            document
        ) in scored_docs:

            if recipe_time is not None:

                if recipe_time <= requested_time:

                    time_filtered_docs.append(
                        document
                    )


        if time_filtered_docs:

            ranked_docs = time_filtered_docs[:4]


        else:

            ranked_docs = [

                document

                for (
                    score,
                    recipe_time,
                    document
                ) in scored_docs[:4]

            ]


    else:

        ranked_docs = [

            document

            for (
                score,
                recipe_time,
                document
            ) in scored_docs[:4]

        ]


    # ========================================================
    # USER INGREDIENTS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🥕 Your Ingredients'
        '</div>',
        unsafe_allow_html=True
    )


    if user_ingredients:

        st.write(
            ", ".join(
                user_ingredients
            )
        )

    else:

        st.write(
            "No specific ingredients detected."
        )


    # ========================================================
    # MISSING INGREDIENTS
    # ========================================================

    missing_ingredients = []


    if ranked_docs:

        best_recipe = ranked_docs[0]


        recipe_ingredients = get_recipe_ingredients(

            best_recipe.page_content

        )


        missing_ingredients = find_missing_ingredients(

            user_ingredients,

            recipe_ingredients

        )


        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # YOU HAVE
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "✅ You Have"
            )


            matching_ingredients = [

                ingredient

                for ingredient in user_ingredients

                if ingredient in recipe_ingredients

            ]


            if matching_ingredients:

                for ingredient in matching_ingredients:

                    st.write(
                        f"✓ {ingredient.title()}"
                    )

            else:

                st.write(
                    "None of your detected ingredients "
                    "are required by this recipe."
                )


        # ----------------------------------------------------
        # YOU MAY NEED
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "🛒 You May Need"
            )


            if missing_ingredients:

                for ingredient in missing_ingredients:

                    st.write(
                        f"• {ingredient.title()}"
                    )

            else:

                st.write(
                    "🎉 You have all the ingredients!"
                )


    # ========================================================
    # SHOPPING LIST
    # ========================================================

    if missing_ingredients:

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🛒 Shopping List'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Ingredients you need to buy:"
        )


        shopping_list = ""


        for ingredient in missing_ingredients:

            st.checkbox(

                ingredient.title(),

                key=f"shopping_{ingredient}"
            )


            shopping_list += (

                f"☐ {ingredient.title()}\n"

            )


        st.write(

            f"**{len(missing_ingredients)} "
            "ingredient(s) needed**"

        )


        st.download_button(

            label="📥 Download Shopping List",

            data=shopping_list,

            file_name="smartchef_shopping_list.txt",

            mime="text/plain",

            use_container_width=True

        )


    # ========================================================
    # SUBSTITUTIONS
    # ========================================================

    if substitution_question:

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🔄 Ingredient Substitutions'
            '</div>',
            unsafe_allow_html=True
        )


        substitution_found = False


        for ingredient in INGREDIENTS:

            if ingredient in question.lower():

                alternatives = find_substitution(
                    ingredient
                )


                if alternatives:

                    substitution_found = True


                    st.write(
                        f"### {ingredient.title()}"
                    )


                    for alternative in alternatives:

                        st.write(
                            f"• {alternative}"
                        )


        if not substitution_found:

            st.info(
                "I don't have a substitution "
                "for that ingredient in my knowledge base."
            )


    # ========================================================
    # RAG CONTEXT
    # ========================================================

    context = "\n\n".join(

        document.page_content

        for document in ranked_docs

    )


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""

You are SmartChef AI, a recipe assistant.

Use ONLY the recipes provided below.

IMPORTANT RULES:

1. Do NOT invent recipes.

2. Prefer recipes matching the user's ingredients.

3. Do not claim the user has ingredients
that they did not mention.

4. If a required ingredient is missing,
mention it.

5. Respect the requested cooking time.

6. If no recipe fits the requirements,
say so honestly.

7. If the user asks for a substitution,
use only the substitution information provided below.

8. Do not invent substitutions.

9. If the shopping list is provided,
do not add ingredients that are not in that list.


USER QUESTION:

{question}


USER INGREDIENTS:

{", ".join(user_ingredients)}


REQUESTED MAXIMUM COOKING TIME:

{
    requested_time
    if requested_time is not None
    else "Not specified"
}


MISSING INGREDIENTS:

{", ".join(missing_ingredients)}


RECIPE DATABASE:

{context}


SUBSTITUTION INFORMATION:

"""


    # ========================================================
    # SUBSTITUTION INFORMATION
    # ========================================================

    if substitution_question:

        for ingredient in INGREDIENTS:

            if ingredient in question.lower():

                alternatives = find_substitution(
                    ingredient
                )


                if alternatives:

                    prompt += (

                        f"\n{ingredient.title()} "
                        "can be substituted with: "

                        f"{', '.join(alternatives)}."

                    )


    # ========================================================
    # FINAL FORMAT
    # ========================================================

    prompt += """

Give the answer in this format:

### 🍽️ Recommended Recipe

**Recipe:** recipe name

**Why it matches:** explain briefly

**Cooking Time:** time

**You have:**
- ingredients the user mentioned

**You may need:**
- missing ingredients

### 🔄 Substitution

If the user asked for a substitution,
mention the available substitution.

### 👨‍🍳 Instructions

Give the basic instructions from the recipe.

Keep the answer simple and useful.

"""


    # ========================================================
    # GEMINI
    # ========================================================

    with st.spinner(
        "👨‍🍳 SmartChef is preparing your answer..."
    ):

        response = llm.invoke(
            prompt
        )


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        "👨‍🍳 SmartChef's Recommendation"
        '</div>',
        unsafe_allow_html=True
    )


    if isinstance(
        response.content,
        list
    ):

        answer = ""


        for block in response.content:

            if (

                isinstance(
                    block,
                    dict
                )

                and block.get("type") == "text"

            ):

                answer += block.get(
                    "text",
                    ""
                )


            elif isinstance(
                block,
                str
            ):

                answer += block


        st.markdown(
            answer
        )


    else:

        st.markdown(
            response.content
        )


    # ========================================================
    # SOURCES
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📚 Sources Used'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "These recipes were retrieved from "
        "the SmartChef knowledge base."
    )


    for i, document in enumerate(

        ranked_docs,

        start=1

    ):

        recipe_text = document.page_content


        recipe_time = get_recipe_time(
            recipe_text
        )


        source_name = document.metadata.get(
            "source",
            "recipes.txt"
        )


        # ----------------------------------------------------
        # RECIPE NAME
        # ----------------------------------------------------

        recipe_name = "Recipe"


        lines = recipe_text.splitlines()


        for line in lines:

            line = line.strip()


            if line:

                recipe_name = line

                break


        # ----------------------------------------------------
        # SOURCE EXPANDER
        # ----------------------------------------------------

        with st.expander(

            f"🍽️ {recipe_name}"

        ):

            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    f"**📄 Source:** "
                    f"{source_name}"
                )


            with col2:

                if recipe_time:

                    st.write(
                        f"**⏱️ Cooking Time:** "
                        f"{recipe_time} minutes"
                    )

                else:

                    st.write(
                        "**⏱️ Cooking Time:** "
                        "Not available"
                    )


            st.divider()


            st.write(
                "**Recipe information retrieved:**"
            )


            st.write(
                recipe_text
            )


# ============================================================
# HOW SMARTCHEF WORKS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🧠 How SmartChef Works'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "SmartChef uses a Retrieval Augmented Generation "
    "(RAG) pipeline to answer recipe-related questions."
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.info(
        "1️⃣ ASK\n\n"
        "Tell SmartChef your ingredients "
        "or cooking requirements."
    )


with col2:

    st.info(
        "2️⃣ RETRIEVE\n\n"
        "FAISS searches the recipe "
        "knowledge base."
    )


with col3:

    st.info(
        "3️⃣ GENERATE\n\n"
        "Gemini uses the retrieved "
        "recipe information."
    )


with col4:

    st.info(
        "4️⃣ ANSWER\n\n"
        "SmartChef provides a useful "
        "recipe recommendation."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🍳 SmartChef AI | "
    "RAG + FAISS + HuggingFace Embeddings + Gemini"
)

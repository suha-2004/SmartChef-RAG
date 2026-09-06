# 👨‍🍳 Smart Chef — AI Recipe Assistant

An AI-powered **Recipe Assistant** that uses **Retrieval-Augmented Generation (RAG)** to provide intelligent, context-aware cooking assistance.

🌐 **Live Demo:** https://smartchef-rag-app.streamlit.app/

---

## 📌 Overview

**Smart Chef** is an AI-powered cooking assistant designed to help users find recipes, understand ingredients, and get useful cooking guidance through a conversational interface.

The project demonstrates the use of **Retrieval-Augmented Generation (RAG)**, where relevant information is retrieved from a knowledge source and provided to an AI model before generating the final response.

Instead of relying only on the model's pre-trained knowledge, the system retrieves relevant recipe information and uses it as context to generate more grounded responses.

---

# ✨ Features

### 🍲 Recipe Assistance

Users can ask questions about recipes, ingredients, cooking methods, and preparation steps.

### 🤖 AI-Powered Responses

The application uses an AI model to generate natural-language responses based on the user's query and retrieved information.

### 🔎 Retrieval-Augmented Generation

Relevant information is retrieved from the recipe knowledge base before generating the response.

This helps the system provide responses based on the available recipe information.

### 🧠 Context-Aware Assistance

The assistant can use retrieved context to provide more relevant answers instead of generating responses without reference information.

### 💬 Interactive Chat Interface

Users can interact with Smart Chef through a simple conversational interface.

### 🌐 Streamlit Web Application

The complete application is deployed as an interactive web application using Streamlit.

---

# 🧠 What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

It combines two main processes:

```text
Retrieval
    +
Generation
    ↓
RAG
```

The retrieval component finds relevant information from a knowledge base.

The generation component uses that retrieved information as context to produce the final answer.

### Simple RAG Workflow

```text
User Question
      ↓
Query Processing
      ↓
Search Knowledge Base
      ↓
Retrieve Relevant Information
      ↓
Provide Context to LLM
      ↓
Generate Answer
      ↓
Display Response
```

---

# 🔄 Smart Chef Workflow

```text
                ┌─────────────────┐
                │   User Query    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Query Processing│
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Knowledge Base  │
                │    Retrieval    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Relevant Recipe │
                │     Context     │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │      LLM        │
                │    Generation   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ AI Response     │
                └─────────────────┘
```

---

# 🧩 RAG Architecture

The application can be understood as two major components.

## 1. Retrieval

The retrieval component searches the recipe knowledge base for information relevant to the user's question.

For example:

```text
User:
"How can I make chicken biryani?"
          ↓
Retrieve relevant recipe information
          ↓
Chicken Biryani recipe
ingredients
preparation steps
cooking instructions
```

---

## 2. Generation

The retrieved information is then provided to the language model as context.

```text
User Query
     +
Retrieved Context
     ↓
Language Model
     ↓
Generated Response
```

This allows the assistant to produce a response grounded in the retrieved recipe information.

---

# 📚 Knowledge Base

The system uses recipe-related information as its knowledge source.

The knowledge base can contain information such as:

* Recipe names
* Ingredients
* Cooking instructions
* Preparation methods
* Cooking time
* Serving information
* Recipe descriptions

The knowledge is retrieved based on the user's query.

---

# 🔍 Example Interaction

### User

```text
How do I make a simple pasta?
```

### Smart Chef

```text
The assistant retrieves relevant pasta recipe information
and generates a response containing ingredients,
preparation steps and cooking instructions.
```

Another example:

```text
User:
"What ingredients are needed for biryani?"

        ↓

Retriever

        ↓

Relevant biryani information

        ↓

LLM

        ↓

Ingredient-focused response
```

---

# 🛠️ Technologies Used

## Programming Language

* **Python**

## AI / Machine Learning

* Large Language Model (LLM)
* Retrieval-Augmented Generation (RAG)
* Natural Language Processing

## Data Processing

* Pandas
* NumPy

## Web Application

* **Streamlit**

## Deployment

* **Streamlit Community Cloud**

---

# 🏗️ Application Architecture

```text
┌───────────────────────────────────────────────┐
│                  User                         │
│                                               │
│          "Give me a pasta recipe"             │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│              Streamlit UI                     │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│             Query Processing                  │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│             Retriever                         │
│                                               │
│       Searches recipe knowledge base           │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│          Relevant Recipe Context              │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│                  LLM                          │
│                                               │
│       Generates context-aware answer          │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│              Smart Chef Response              │
└───────────────────────────────────────────────┘
```

---

# 📂 Project Structure

```text
SmartChef/
│
├── app.py
│
├── data/
│   └── recipe knowledge base
│
├── models/
│   └── model / embedding configuration
│
├── utils/
│   └── helper functions
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> The exact folder structure may vary depending on the current version of the project.

---

# 💻 Installation & Setup

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

## 2. Navigate to the Project

```bash
cd SmartChef
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Configure Environment Variables

If the application requires an API key for the language model or other external services, create a `.env` file and add the required credentials.

Example:

```env
API_KEY=your_api_key
```

> Never commit API keys or other secrets to GitHub.

## 7. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🌐 Live Demo

Try the deployed Smart Chef application:

## 🚀 https://smartchef-rag-app.streamlit.app/

The application provides an interactive interface where users can communicate with the AI cooking assistant.

---

# 🎯 Project Objectives

The main objectives of Smart Chef are:

* To build an AI-powered recipe assistant
* To understand Retrieval-Augmented Generation
* To combine information retrieval with LLM generation
* To create a domain-specific AI assistant
* To provide context-aware recipe assistance
* To build an interactive AI web application
* To deploy an AI application using Streamlit

---

# 🧠 Key Concepts Demonstrated

This project demonstrates practical concepts in:

### Artificial Intelligence

* Large Language Models
* Generative AI
* Natural Language Processing

### RAG

* Knowledge retrieval
* Context augmentation
* Grounded generation
* Domain-specific question answering

### Application Development

* Python
* Streamlit
* Interactive UI
* API integration

### Deployment

* Git
* GitHub
* Streamlit Community Cloud

---

# 💡 Why Use RAG?

A standard LLM generates answers primarily from the knowledge learned during training.

A RAG system adds an additional retrieval step:

```text
Traditional LLM

User Question
     ↓
    LLM
     ↓
Answer
```

With RAG:

```text
User Question
     ↓
Retrieve Relevant Information
     ↓
Retrieved Context
     ↓
    LLM
     ↓
Context-Aware Answer
```

This approach is useful when the application needs to answer questions using a specific knowledge base.

---

# 🔮 Future Improvements

Smart Chef can be extended with additional features such as:

* 🥗 Personalized recipes based on available ingredients
* 🧑‍🍳 Step-by-step cooking mode
* 🛒 Automatic grocery list generation
* 🍽️ Personalized meal planning
* 🥦 Dietary preference filtering
* 🌱 Vegetarian and vegan recipe recommendations
* 🌶️ Spice-level customization
* ⏱️ Cooking-time based recommendations
* 📊 Nutritional information
* 🗣️ Voice-based interaction
* 📷 Image-based ingredient recognition
* 🌍 Multi-language recipe assistance
* 💾 Save favorite recipes
* 👤 User profiles and personalized recommendations

---

# ⚠️ Disclaimer

Smart Chef is intended for **educational and informational purposes**.

Recipe information and AI-generated responses should be reviewed carefully. Cooking results can vary depending on ingredients, equipment, measurements, and preparation methods.

Users should use appropriate food-safety practices when preparing food.

---

# 👩‍💻 Author

## Suha

**B.Tech Computer Science & Engineering**

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Generative AI
* Large Language Models
* Retrieval-Augmented Generation
* Full-Stack Development

---

# 🔗 Project Links

| Resource            | Link                                            |
| ------------------- | ----------------------------------------------- |
| 🌐 Live Application | https://smartchef-rag-app.streamlit.app/        |
| 🐙 GitHub           | https://github.com/suha-2004                    |
| 💼 LinkedIn         | https://www.linkedin.com/in/suha-i-a-a769ab358/ |

---

# ⭐ Support

If you found this project interesting, consider giving the repository a ⭐ on GitHub.

Thank you for checking out **Smart Chef — AI Recipe Assistant!** 👨‍🍳🤖

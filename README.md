# Rag-Chat-Bot

Rag-Chat-Bot is a Retrieval-Augmented Generation (RAG) chatbot that uses OpenAI's GPT-4 and embeddings to answer user queries based on a custom chat dataset.

## Features

- Loads chat data from CSV files
- Stores chat embeddings in a persistent ChromaDB collection
- Retrieves relevant chats using semantic search
- Generates responses using GPT-4, strictly based on retrieved context


## Setup

1. **Clone the repository** and open in VS Code (recommended).
2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
3. **Set your OpenAI API key**:
    - Create a `.env` file in the project root:
      ```
      OPENAI_API_KEY=your_openai_api_key
      ```
4. **Run the chatbot**:
    ```sh
    python [chatbot.py](http://_vscodecontentref_/4)
    ```

## Usage

- The chatbot loads chat data from `data/chat_training.csv` by default.
- Enter your query at the prompt. Type `exit` or press Enter to quit.

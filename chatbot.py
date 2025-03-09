import csv
from openai import OpenAI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

def read_chats():
    dict={}
    with open('chat_training_2.csv', mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            dict[row[1]]=row[4]
    return dict

def get_embedding(text):
    response = client.embeddings.create(
        input=text, model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def store_chats(chats):
    n=0
    length=len(chats)
    
    for chat_id, text in chats.items():
        embedding = get_embedding(text)
        collection.add(
            ids=[chat_id],
            embeddings=[embedding],
            documents=[text]
        )
        
        n+=1
        if n % 100 == 0:
            print("----------------------------------")
            print(f"Stored chat {n}/{length}")

def retrieve_top_chats(query, top_k=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0] if results["documents"] else []

def generate_response(query, relevant_docs):
    context = "\n\n".join(relevant_docs)
    prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a chatbot trying to help users with issus. ONLY USE THE INFORMATION PROVIDED IN THE CHAT."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    chats = read_chats()
    store_chats(chats)

    while True:
        os.system('clear')

        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit" or query.lower() == "":
            break
        
        relevant_docs = retrieve_top_chats(query, top_k=3)
        response = generate_response(query, relevant_docs)
        print("\nResponse:", response)
        input("\nPress Enter to continue...")

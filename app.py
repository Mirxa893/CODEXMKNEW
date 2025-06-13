import gradio as gr
import requests
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# ✅ Load RAG-related files
with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

index = faiss.read_index("faiss_index.bin")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Use your OpenRouter API key from environment
API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-chat-v3-0324:free"

# ✅ Function to search relevant context
def get_context(query, top_k=5):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return "\n".join([texts[i] for i in I[0]])

# ✅ Function to handle chat

def chat_fn(message, history):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    context = get_context(message)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following context to answer: " + context}
    ]

    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL,
        "messages": messages
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"❌ Error: {e}"

    return reply

# ✅ Launch Gradio Chat Interface
gr.ChatInterface(
    fn=chat_fn,
    title="CODEX MIRXA KAMRAN",
    description="Chat with AI MODEL trained By Mirxa Kamran",
    theme="soft"
).launch()
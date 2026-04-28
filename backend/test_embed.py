import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

models = [
    "models/embedding-001",
    "models/text-embedding-004",
    "text-embedding-004",
    "embedding-001"
]

for model in models:
    print(f"Trying model: {model}")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=model)
        result = embeddings.embed_query("hello")
        print(f"SUCCESS: {model} works! Dimension: {len(result)}")
        break
    except Exception as e:
        print(f"FAILED: {model} -> {e}")

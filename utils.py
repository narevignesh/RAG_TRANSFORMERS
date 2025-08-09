# utils.py
import os
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss as lc_faiss

def preload_sentence_transformer(model_name: str):
    # Preload model weights onto CPU — helps avoid meta tensor issues
    try:
        print(f"[utils] Preloading SentenceTransformer '{model_name}' to CPU...")
        SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        print(f"[utils] Warning: preloading model failed: {e}. Proceeding — may still work.")

def load_vectorstore(index_dir="data/faiss_index", model_name=None):
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    preload_sentence_transformer(model_name)

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    # safer: only allow deserialization if this is your local index you created
    vs = lc_faiss.FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return vs

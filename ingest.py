# ingest.py
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss as lc_faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

# Default configs (can be overridden by .env)
DOC_URL = os.getenv(
    "DOC_URL",
    "DOC_URL2",
    "https://huggingface-transformers.readthedocs.io/en/latest/"
)
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")


def preload_sentence_transformer(model_name: str):
    """
    Force load the sentence-transformers model to CPU to avoid meta-tensor errors.
    """
    print(f"[ingest] Preloading SentenceTransformer '{model_name}' on CPU...")
    SentenceTransformer(model_name, device="cpu")
    print("[ingest] Preload finished.")


def ingest_and_save(doc_url: str, model_name: str, index_dir: str):
    """
    Load documents from the given URL, split them, embed them, and save to FAISS index.
    """
    os.makedirs(index_dir, exist_ok=True)

    # 1) Load the docs
    print(f"[ingest] Loading pages from: {doc_url}")
    loader = WebBaseLoader(doc_url)
    docs = loader.load()
    print(f"[ingest] Loaded {len(docs)} document(s).")

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("[ingest] Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks.")

    # 3) Preload model to CPU to avoid meta tensor problems
    preload_sentence_transformer(model_name)

    # 4) Create HuggingFaceEmbeddings wrapper
    print("[ingest] Initializing embeddings wrapper...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )

    # 5) Build FAISS index
    print("[ingest] Building FAISS index...")
    vectorstore = lc_faiss.FAISS.from_documents(chunks, embeddings)

    # 6) Save FAISS locally
    print(f"[ingest] Saving FAISS index to: {index_dir}")
    vectorstore.save_local(index_dir)
    print("[ingest] Done. Index saved.")


def main():
    ingest_and_save(DOC_URL, MODEL_NAME, INDEX_DIR)


if __name__ == "__main__":
    main()

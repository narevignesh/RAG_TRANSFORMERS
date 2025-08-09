# RAG Chat — Transformers docs

## Setup
1. Copy `.env.example` → `.env` and fill keys.
2. Install dependencies:
   pip install -r requirements.txt

3. Build the FAISS index (run once):
   python ingest.py

   If you get meta-tensor errors, confirm EMBEDDING_MODEL is a small model (default: all-MiniLM-L6-v2).

4. Start the Streamlit app:
   streamlit run app.py  or python -m stramlit run

Open http://localhost:8501

## Notes
- Do NOT commit .env or your keys.
- If FAISS is corrupted/old, delete `data/faiss_index` and re-run `ingest.py`.
- If you see "Cannot copy out of meta tensor", make sure `sentence-transformers` model preloads to CPU (ingest.py & utils.py handle that).

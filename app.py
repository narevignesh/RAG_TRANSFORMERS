# app.py
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_groq.chat_models import ChatGroq
from utils import load_vectorstore

# --- UI SETTINGS ---
st.set_page_config(page_title="RAG Chat — Transformers docs", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #1F77B4;'>🤖 RAG Chat — Transformers</h1>",
    unsafe_allow_html=True
)

# --- Config ---
# Updated website source
DOC_URL = os.getenv("DOC_URL", "https://huggingface-transformers.readthedocs.io/en/latest/")
INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "RAG Chat — Transformers RTD")
os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME

# --- Load vectorstore using new source URL ---
@st.cache_resource(show_spinner=True)
def get_vectorstore(index_dir, model_name, doc_url):
    # ingest.py or utils must be updated to accept doc_url
    from ingest import ingest_and_save  # We'll create helper
    ingest_and_save(doc_url, model_name, index_dir)
    return load_vectorstore(index_dir=index_dir, model_name=model_name)

try:
    vectorstore = get_vectorstore(INDEX_DIR, EMBEDDING_MODEL, DOC_URL)
except Exception as e:
    st.error(f"Failed to load vectorstore: {e}")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- LLM ---
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not set in .env.")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# --- Memory (session state) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create memory object
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# --- Prompt + Parser ---
response_schemas = [
    ResponseSchema(name="answer", description="Assistant's answer."),
    ResponseSchema(name="evidence_summary", description="One-sentence summary of evidence (optional).")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

prompt_template = """
You are an expert assistant answering questions from the Hugging Face Transformers documentation.
Use ONLY the provided context. If not available, say "I don't know".

{format_instructions}

Context:
{context}

Question:
{question}

Answer in JSON format:
"""
prompt = PromptTemplate(
    input_variables=["context", "question", "format_instructions"],
    template=prompt_template
)

# --- Conversational Retrieval Chain with memory ---
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# --- User Input UI ---
st.markdown("### 💬 Ask about Transformers")
question = st.text_input("", placeholder="Type your question here...")
if st.button("🚀 Send", use_container_width=True) and question:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(question)
        joined_context = "\n\n".join(
            [f"{d.page_content}\n(Source: {d.metadata.get('source','unknown')})" for d in docs[:6]]
        )
        filled_prompt = prompt.format(
            context=joined_context,
            question=question,
            format_instructions=format_instructions
        )
        result = chain({"question": question})
        raw_answer = result.get("answer", "")
        source_documents = result.get("source_documents", [])

        try:
            parsed = parser.parse(raw_answer)
        except:
            parsed = {"answer": raw_answer, "evidence_summary": ""}

        # Save to session
        st.session_state.chat_history.append(("User", question))
        st.session_state.chat_history.append(("Assistant", parsed.get("answer", raw_answer)))

    st.success("Done!")
    st.subheader("**Answer**")
    st.info(parsed.get("answer", raw_answer))

    if parsed.get("evidence_summary"):
        st.markdown(f"*Evidence summary:* {parsed['evidence_summary']}")

    if source_documents:
        st.subheader("Top sources used")
        for i, d in enumerate(source_documents[:5], 1):
            src = d.metadata.get("source", "unknown")
            snippet = d.page_content[:300].replace("\n", " ")
            st.markdown(f"**{i}.** `{src}`")
            st.code(snippet + ("..." if len(d.page_content) > 300 else ""))

# --- Conversation History ---
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🕒 Conversation History This Session")
    for speaker, text in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**👤 {speaker}:** {text}")
        else:
            st.info(text)

# lab4.py — Lab 4 RAG chatbot with ChromaDB (IN-MEMORY VERSION)

import os
import glob
import textwrap
from typing import List, Dict

import streamlit as st
from openai import OpenAI

# SQLite fix
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import PyPDF2
import chromadb
from chromadb.utils import embedding_functions

# ===================== Constants =====================
PAGE_TITLE        = "Lab 4 — RAG chatbot with ChromaDB"
PDF_FOLDER        = "lab4_pdfs"
CHROMA_COLLECTION = "Lab4Collection"
EMBED_MODEL       = "text-embedding-3-small"
TOP_K             = 3
BUFFER_PAIRS      = 6
MODEL_NAME        = "gpt-4o-mini"

# ===================== UI Header =====================
st.title(PAGE_TITLE)
st.caption("This page builds a ChromaDB from PDFs and uses RAG to answer questions.")

# ===================== Utilities =====================
def ensure_pdf_folder() -> List[str]:
    """Verify the PDF folder exists and return a list of PDF paths."""
    if not os.path.isdir(PDF_FOLDER):
        os.makedirs(PDF_FOLDER, exist_ok=True)
        st.error(f"Created PDF folder: {os.path.abspath(PDF_FOLDER)}")
        st.info("Please place your PDF files in this folder and rerun.")
        st.stop()
    
    pdfs = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    if not pdfs:
        st.warning(f"No PDFs found in {PDF_FOLDER}. Add files and rerun.")
        st.stop()
    return pdfs

def read_pdf_to_text(path: str) -> str:
    """Extract plain text from a PDF file."""
    parts = []
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
                except:
                    pass
    except Exception as e:
        st.warning(f"Error reading {os.path.basename(path)}: {e}")
    return "\n".join(parts).strip()

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    """Simple char-based chunking."""
    text = " ".join(text.split())
    chunks = []
    for i in range(0, len(text), max_chars):
        chunk = text[i:i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ===================== Build ChromaDB (IN-MEMORY) =====================
@st.cache_resource(show_spinner=True)
def build_chromadb():
    """Create an in-memory Chroma collection with OpenAI embeddings."""
    
    # Get API key
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()
    
    # Use IN-MEMORY client instead of persistent
    client = chromadb.Client()
    
    # Create embedding function
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )
    
    # Create collection
    coll = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef
    )
    
    # Index PDFs
    st.sidebar.info("Indexing PDFs...")
    pdf_paths = ensure_pdf_folder()
    
    ids, docs, metas = [], [], []
    
    for pdf_path in pdf_paths:
        fname = os.path.basename(pdf_path)
        text = read_pdf_to_text(pdf_path)
        
        if not text:
            continue
        
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{fname}-{idx:04d}")
            docs.append(chunk)
            metas.append({"filename": fname, "chunk": idx})
    
    if ids:
        # Add in batches
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            coll.add(
                documents=docs[i:batch_end],
                metadatas=metas[i:batch_end],
                ids=ids[i:batch_end]
            )
        st.sidebar.success(f"Indexed {len(ids)} chunks from {len(pdf_paths)} PDFs")
    
    return coll

# Build collection
if "Lab4_vectorDB" not in st.session_state:
    st.session_state.Lab4_vectorDB = build_chromadb()

# ===================== Retrieval =====================
def retrieve_context(query: str, k: int = TOP_K) -> str:
    """Retrieve context from vector DB."""
    try:
        coll = st.session_state.Lab4_vectorDB
        results = coll.query(query_texts=[query], n_results=k)
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        context_parts = []
        for doc, meta in zip(docs, metas):
            if doc:
                fname = meta.get("filename", "unknown") if meta else "unknown"
                snippet = textwrap.shorten(doc.replace("\n", " "), width=480, placeholder="...")
                context_parts.append(f"[{fname}] {snippet}")
        
        if not context_parts:
            return ""
        
        return (
            "COURSE PDF CONTEXT (retrieved):\n" +
            "\n".join(context_parts) + "\n\n" +
            "Use the above context if helpful. Mention which filenames informed your answer."
        )
    except Exception as e:
        return f"(Retrieval error: {e})"

# ===================== OpenAI client =====================
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

# ===================== Chat =====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about the course PDFs."}
    ]

# Controls
st.sidebar.header("Options")
buffer_pairs = st.sidebar.slider("Buffer size", 1, 10, BUFFER_PAIRS)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about the course PDFs."}
    ]
    st.rerun()

def get_buffered(messages, pairs):
    if len(messages) <= 2:
        return messages
    head = [messages[0]]
    tail_size = pairs * 2
    tail = messages[-tail_size:] if len(messages) > tail_size + 1 else messages[1:]
    return head + tail

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_text := st.chat_input("Ask about the PDFs..."):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    
    # Get context
    rag_context = retrieve_context(user_text, k=TOP_K)
    
    # Build messages
    system_parts = [
        "You are a helpful teaching assistant.",
        "Answer clearly and mention PDF sources when used."
    ]
    if rag_context:
        system_parts.append(rag_context)
    
    system_msg = {"role": "system", "content": "\n\n".join(system_parts)}
    msgs = [system_msg] + get_buffered(st.session_state.messages, buffer_pairs)
    
    # Generate response
    with st.chat_message("assistant"):
        st.caption(f"Model: {MODEL_NAME}")
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            temperature=0.7,
            stream=True,
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.sidebar.markdown("---")
st.sidebar.metric("Collection", CHROMA_COLLECTION)
st.sidebar.metric("Chunks", st.session_state.Lab4_vectorDB.count())
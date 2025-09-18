# HW3.py  (Upgraded for Lab 4 - RAG with ChromaDB)

import os
import glob
import textwrap
import requests
from typing import List, Dict

import streamlit as st
from bs4 import BeautifulSoup

# --- LLM SDKs ---
from openai import OpenAI
import anthropic
import google.generativeai as genai

# --- Token counting for buffers ---
import tiktoken

# --- PDF + Vector DB (Chroma) ---
# sqlite fix for some environments (esp. Streamlit Cloud)
try:
    __import__("pysqlite3")  # noqa: F401
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import PyPDF2
import chromadb
from chromadb.utils import embedding_functions

# ---------------------- Constants ----------------------
PAGE_TITLE = "HW3 + Lab4 — Streaming Chatbot with URL Context & RAG (ChromaDB)"
PDF_FOLDER = "lab4_pdfs"                 # Put your 7 PDFs here
CHROMA_PATH = "./ChromaDB_for_lab"       # Local persistent dir
CHROMA_COLLECTION = "Lab4Collection"     # Required name per spec
EMBED_MODEL = "text-embedding-3-small"   # OpenAI Embeddings (1536-dim)
TOP_K_RETRIEVE = 3                        # show 3 docs in the test & RAG

# ---------------------- Page & Sidebar ----------------------
st.title(PAGE_TITLE)
st.sidebar.header("Configuration")

# URLs (two inputs)
st.sidebar.markdown("### URL Sources")
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/article-1")
url2 = st.sidebar.text_input("URL 2 (Optional)", placeholder="https://example.com/article-2")

# LLM Provider selection
st.sidebar.markdown("### LLM Selection")
provider = st.sidebar.selectbox("Select LLM Provider", ["OpenAI", "Anthropic", "Google Gemini"])

# Model selection per provider (cheap/expensive)
if provider == "OpenAI":
    model_type = st.sidebar.radio("Model Type", ["Cheap", "Expensive"])
    model_to_use = "gpt-4o-mini" if model_type == "Cheap" else "gpt-4o"
elif provider == "Anthropic":
    model_type = st.sidebar.radio("Model Type", ["Cheap", "Expensive"])
    # If you only have Haiku, use it for both
    model_to_use = "claude-3-haiku-20240307" if model_type == "Cheap" else "claude-3-haiku-20240307"
else:  # Google Gemini
    model_type = st.sidebar.radio("Model Type", ["Cheap", "Expensive"])
    model_to_use = "gemini-1.5-flash" if model_type == "Cheap" else "gemini-1.5-pro"

# Memory type selection
st.sidebar.markdown("### Conversation Memory")
memory_type = st.sidebar.selectbox(
    "Memory Type",
    ["Buffer (6 questions)", "Conversation Summary", "Token Buffer (2000 tokens)"]
)

# Optional: simple Vector DB test UI toggle
show_vectordb_test = st.sidebar.checkbox("Show Vector DB test panel (Lab 4a step 3)", value=True)

# ---------------------- Initialize Clients ----------------------
@st.cache_resource
def get_client(provider_name: str, model_hint: str | None = None):
    """Return the correct client handle based on provider."""
    if provider_name == "OpenAI":
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OPENAI_API_KEY in secrets or environment.")
            return None
        return OpenAI(api_key=api_key)

    if provider_name == "Anthropic":
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Missing ANTHROPIC_API_KEY in secrets or environment.")
            return None
        return anthropic.Anthropic(api_key=api_key)

    # Google
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in secrets or environment.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_hint or "gemini-1.5-flash")

# ---------------------- Helpers: token counting & buffers ----------------------
@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(messages: List[Dict[str, str]]) -> int:
    enc = get_tokenizer()
    total = 0
    for m in messages:
        total += len(enc.encode(m["role"])) + len(enc.encode(m["content"])) + 4
    return total + 3

def get_message_buffered(messages: List[Dict[str, str]], keep_pairs: int) -> List[Dict[str, str]]:
    if len(messages) <= 1:
        return messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    other = [m for m in messages if m["role"] != "system"]
    to_keep = keep_pairs * 2
    other = other[-to_keep:] if len(other) > to_keep else other
    return system_msgs + other

def get_token_buffered(messages: List[Dict[str, str]], max_tok: int) -> List[Dict[str, str]]:
    if not messages:
        return []
    system_msgs = [m for m in messages if m["role"] == "system"]
    other = [m for m in messages if m["role"] != "system"]
    res, total = [], count_tokens(system_msgs)
    for m in reversed(other):
        t = count_tokens([m])
        if total + t <= max_tok:
            res.insert(0, m); total += t
        else:
            break
    return system_msgs + res

# ---------------------- Helpers: URL loading ----------------------
@st.cache_data(show_spinner=False)
def read_url_content(url: str) -> str | None:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        txt = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in txt.splitlines()]
        txt = "\n".join([ln for ln in lines if ln])
        return txt[:10000]
    except Exception as e:
        st.sidebar.error(f"Read URL failed: {e}")
        return None

# ---------------------- Lab 4a: ChromaDB build & store ----------------------
def ensure_pdf_folder() -> List[str]:
    """Verify the PDF folder exists and return a list of PDF paths."""
    if not os.path.isdir(PDF_FOLDER):
        st.error(f"PDF folder not found: {os.path.abspath(PDF_FOLDER)}")
        st.info("Create the folder and place your 7 PDFs inside, then rerun.")
        st.stop()
    pdfs = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    if not pdfs:
        st.warning(f"No PDFs found in {PDF_FOLDER}. Add files and rerun.")
    return pdfs

def read_pdf_to_text(path: str) -> str:
    """Read a PDF to plain text using PyPDF2."""
    out = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                pass
    return "\n".join(out).strip()

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    """Simple char-based chunking to keep chunks short for retrieval."""
    text = " ".join(text.split())
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

@st.cache_resource(show_spinner=True)
def build_or_load_chromadb() -> chromadb.Collection:
    """Create/load the ChromaDB collection and store it in session state once per run."""
    # 1) Init Chroma persistent client
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 2) Create OpenAI embedding function (will be used for add/query consistently)
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY for embeddings.")
        st.stop()
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    # 3) Get/create collection with embedding_function
    coll = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=ef)

    # If empty, build from PDFs
    count = coll.count()
    if count == 0:
        pdf_paths = ensure_pdf_folder()
        ids, docs, metas = [], [], []
        for p in pdf_paths:
            fname = os.path.basename(p)
            text = read_pdf_to_text(p)
            if not text:
                continue
            for idx, chunk in enumerate(chunk_text(text)):
                ids.append(f"{fname}-{idx}")
                docs.append(chunk)
                metas.append({"filename": fname, "chunk": idx})
        if ids:
            coll.add(documents=docs, metadatas=metas, ids=ids)
    return coll

# build once per run and store handle
if "Lab4_vectorDB" not in st.session_state:
    st.session_state.Lab4_vectorDB = build_or_load_chromadb()

# ---------------------- Vector DB Test Panel (Lab 4a step 3) ----------------------
if show_vectordb_test:
    st.subheader("Test the Vector DB (Simple Search)")
    q = st.text_input("Enter a query to test (e.g., 'Generative AI', 'Text Mining', 'Data Science Overview')")
    if st.button("Search Top 3") and q:
        try:
            res = st.session_state.Lab4_vectorDB.query(query_texts=[q], n_results=TOP_K_RETRIEVE)
            hits = list(zip(res.get("ids", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0] if "distances" in res else []))
            if not hits:
                st.info("No results.")
            else:
                st.write("Top results (by filename):")
                for _id, meta, *_ in hits:
                    st.write(f"• {meta.get('filename')}")
        except Exception as e:
            st.error(f"Query failed: {e}")

# ---------------------- URL Loading & Context ----------------------
url1_text = read_url_content(url1) if url1 else None
url2_text = read_url_content(url2) if url2 else None

if url1_text or url2_text:
    combined = "You have access to the following web content. Use it to answer questions:\n\n"
    if url1_text:
        combined += f"=== Content from URL 1: {url1} ===\n{url1_text}\n\n"
    if url2_text:
        combined += f"=== Content from URL 2: {url2} ===\n{url2_text}\n\n"
    combined += "Please use this content to answer the user's questions. Cite which URL source you're using when relevant."
    st.session_state["url_context"] = combined
else:
    st.session_state["url_context"] = ""

ready = []
if url1_text: ready.append("URL 1")
if url2_text: ready.append("URL 2")
if ready:
    st.sidebar.success(f"✅ Loaded: {', '.join(ready)}")

# ---------------------- Conversation State ----------------------
if "conv_summary" not in st.session_state:
    st.session_state.conv_summary = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can answer using your URLs and our course PDFs. Ask me anything!"}
    ]

if st.sidebar.button("Clear Conversation 🗑️"):
    st.session_state.conv_summary = ""
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can answer using your URLs and our course PDFs. Ask me anything!"}
    ]
    st.rerun()

# ---------------------- Show History ----------------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# ---------------------- Summary updater (for memory type) ----------------------
def update_summary(old_summary: str, last_turn: list, provider_name: str) -> str:
    """Update running conversation summary (short)."""
    try:
        prompt = f"""Update a running summary for later recall.
OLD SUMMARY:
{old_summary}

NEW TURN:
User: {last_turn[0]["content"]}
Assistant: {last_turn[1]["content"]}

Return a concise updated summary (<=120 words)."""
        if provider_name == "OpenAI":
            client = get_client("OpenAI")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Lightweight fallback
            return (old_summary + "\n" + f"User asked about: {last_turn[0]['content'][:60]}...").strip()
    except Exception:
        return old_summary

# ---------------------- RAG: retrieve from Chroma for a query ----------------------
def retrieve_from_vectordb(query: str, k: int = TOP_K_RETRIEVE) -> str:
    """Return a compact context block from top-k retrieved chunks."""
    try:
        res = st.session_state.Lab4_vectorDB.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        pairs = []
        for d, m in zip(docs, metas):
            fname = (m or {}).get("filename", "unknown.pdf")
            snippet = textwrap.shorten(d.replace("\n", " "), width=500, placeholder="…")
            pairs.append(f"[{fname}] {snippet}")
        if not pairs:
            return ""
        joined = "\n".join(pairs)
        return f"COURSE PDF CONTEXT (retrieved):\n{joined}\n\nUse the above context if helpful."
    except Exception as e:
        return f"(Retrieval error: {e})"

# ---------------------- Generation (streaming per provider) ----------------------
def generate_streaming_response(messages, provider_name, model):
    client = get_client(provider_name, model)
    if not client:
        return "Error: Client not initialized. Check API keys."

    if provider_name == "OpenAI":
        stream = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0.7)
        return st.write_stream(stream)

    if provider_name == "Anthropic":
        sys_content = ""
        claude_msgs = []
        for m in messages:
            if m["role"] == "system":
                sys_content = (sys_content + "\n\n" + m["content"]) if sys_content else m["content"]
            elif m["role"] in ["user", "assistant"]:
                claude_msgs.append({"role": m["role"], "content": m["content"]})
        if not claude_msgs or claude_msgs[0]["role"] != "user":
            claude_msgs.insert(0, {"role": "user", "content": "Hello"})
        try:
            with client.messages.stream(
                model=model,
                messages=claude_msgs,
                system=sys_content or None,
                max_tokens=2000,
                temperature=0.7
            ) as stream:
                buf = ""
                out = st.empty()
                for txt in stream.text_stream:
                    buf += txt
                    out.markdown(buf)
                return buf
        except Exception as e:
            return f"Anthropic API error: {e}"

    # Google Gemini (build one big prompt)
    try:
        full_prompt = ""
        for m in messages:
            if m["role"] == "system":
                full_prompt += f"SYSTEM:\n{m['content']}\n\n"
        full_prompt += "CONVERSATION:\n"
        for m in messages:
            if m["role"] != "system":
                full_prompt += f"{m['role'].upper()}: {m['content']}\n"
        full_prompt += "\nASSISTANT:\n"
        resp = client.generate_content(full_prompt, stream=True)
        buf, out = "", st.empty()
        for ch in resp:
            if getattr(ch, "text", None):
                buf += ch.text
                out.markdown(buf)
        return buf
    except Exception as e:
        return f"Google API error: {e}"

# ---------------------- Chat Input & RAG Generation ----------------------
if user_text := st.chat_input("Ask about the URLs and/or course PDFs..."):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Memory strategy
    if memory_type == "Buffer (6 questions)":
        msgs_for_api = get_message_buffered(st.session_state.messages, keep_pairs=6)
    elif memory_type == "Token Buffer (2000 tokens)":
        msgs_for_api = get_token_buffered(st.session_state.messages, max_tok=2000)
    else:
        trimmed = get_message_buffered(st.session_state.messages, keep_pairs=2)
        if st.session_state.conv_summary:
            summary_msg = {"role": "system", "content": "Conversation summary so far:\n" + st.session_state.conv_summary}
            msgs_for_api = [summary_msg] + trimmed
        else:
            msgs_for_api = trimmed

    # Build system context blocks (URL + RAG)
    sys_blocks = []
    if st.session_state.get("url_context"):
        sys_blocks.append(st.session_state["url_context"])

    rag_context = retrieve_from_vectordb(user_text, k=TOP_K_RETRIEVE)
    if rag_context:
        sys_blocks.append(rag_context)

    if sys_blocks:
        msgs_for_api = [{"role": "system", "content": "\n\n".join(sys_blocks)}] + msgs_for_api

    # Generate
    with st.chat_message("assistant"):
        st.caption(f"Using {provider} - {model_to_use}")
        try:
            assistant_text = generate_streaming_response(msgs_for_api, provider, model_to_use)
        except Exception as e:
            assistant_text = f"Error: {e}"
            st.error(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    # Update conversation summary (if selected)
    if memory_type == "Conversation Summary" and len(st.session_state.messages) >= 2:
        last_turn = st.session_state.messages[-2:]
        st.session_state.conv_summary = update_summary(st.session_state.conv_summary, last_turn, provider)

# ---------------------- Sidebar Metrics ----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Statistics")

if memory_type == "Buffer (6 questions)":
    _buf = get_message_buffered(st.session_state.messages, keep_pairs=6)
elif memory_type == "Token Buffer (2000 tokens)":
    _buf = get_token_buffered(st.session_state.messages, max_tok=2000)
else:
    _buf = get_message_buffered(st.session_state.messages, keep_pairs=2)

c1, c2 = st.sidebar.columns(2)
with c1:
    st.metric("Total Messages", len(st.session_state.messages))
with c2:
    st.metric("In Memory", len(_buf))

st.markdown("---")
st.info(
    f"**Config**: Provider={provider} ({model_type}) • Memory={memory_type} • "
    f"URLs Loaded={1 if url1_text else 0 + 1 if url2_text else 0}/2 • "
    f"VectorDB Collection='{CHROMA_COLLECTION}'"
)

# Optional: show a short preview of retrieved PDF chunks for last question
if rag_context:
    with st.expander("🔎 View retrieved PDF snippets used this turn"):
        st.write(rag_context)

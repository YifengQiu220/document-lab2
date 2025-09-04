import os
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

st.title("Lab 2 — PDF Summarizer")

def get_openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key. Please provide `OPENAI_API_KEY` via Streamlit secrets or environment variables.")
        st.stop()
    return OpenAI(api_key=api_key)

if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = get_openai_client()

client: OpenAI = st.session_state["openai_client"]

st.sidebar.header("Options")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

summary_type = st.sidebar.selectbox(
    "Select summary type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ],
)

use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)", value=False)
model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    document_text = "".join([page.get_text() for page in doc])

    st.subheader("Generated Summary")

    prompt = f"Please {summary_type.lower()}:\n\n{document_text}"

    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    st.write_stream(stream)



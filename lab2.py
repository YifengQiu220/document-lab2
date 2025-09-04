

# lab2.py
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

st.title(" Lab 2 — Document Q&A")


st.subheader("Lab2a: Create Multi Page app for all labs")
st.write(
    """
    We can view the Lab1 as well as Lab2 on two different pages
    The main function: PDF Summary LLM is in Lab1
    """
)

st.subheader("Lab 2b: Update the application to use secrets")
st.write(
    """
    In Lab1 page, we do not need to copy API key anymore  
    """
)


st.subheader("Lab 2c: Provide different types of summaries with 4o and 4o mini")
st.write(
    """
    After we upload pdf document, all we need to do is to select mode of summary and LLM models  
    """
)


st.subheader("Lab 2d — Default Model Choice")
st.write(
    """
    The default model is **gpt-4o-mini**.  
    - It provides a good balance between **quality** and **cost**.  
    - For most summarization tasks, `gpt-4o-mini` is fast, cheaper, and still accurate enough.  /  
    """
)
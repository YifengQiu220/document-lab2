# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="IST688 Labs", page_icon="🎓")

pages = {
    "Labs": [
        st.Page("lab1.py", title="Lab 1 — Document Q&A", icon="📄"),
        st.Page("lab2.py", title="Lab 2 — Multi-page App", icon="📊"),
        st.Page("lab3.py", title="Lab 3 — Streaming Chatbot", icon="🤖"), 
        st.Page("lab4.py", title="Lab 4 — Vector DB Test"),  
        st.Page("lab5.py", title="Lab 5 — weather chatbot"),
        st.Page("lab6.py", title="Lab 6 — AI Fact-Checker "),
        st.Page("lab8.py", title="Lab 8 — RAG and Re-Ranking"),
    ],
}

st.navigation(pages).run()
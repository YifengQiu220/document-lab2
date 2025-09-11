# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="IST688 Labs", page_icon="🎓")

pages = {
    "Labs": [
        st.Page("lab1.py", title="Lab 1 — Document Q&A", icon="📄"),
        st.Page("lab2.py", title="Lab 2 — Multi-page App", icon="📊"),
        st.Page("lab3.py", title="Lab 3 — Streaming Chatbot", icon="🤖"),  
    ],
}

st.navigation(pages).run()
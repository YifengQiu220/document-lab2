import streamlit as st
from pypdf import PdfReader

# Try different import patterns for compatibility
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

# Import alternative chain components
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

def format_docs(docs):
    """Format retrieved documents for the prompt."""
    return "\n\n".join([d.page_content for d in docs])

def get_rag_chain(vectorstore, openai_api_key, k_value):
    """
    Create a simple RAG chain using LCEL (LangChain Expression Language).
    """
    
    # Create LLM instance
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, 
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    
    # Create retriever with k parameter for re-ranking
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_value}
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant analyzing SEC 10-Q filings. Use the following context to answer the question:\n\n{context}"),
        ("human", "{question}")
    ])
    
    # Create chain using LCEL
    rag_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | llm
    
    return rag_chain, retriever

# --- Streamlit Application Interface ---

st.title("🧪 Lab 8: RAG and Re-Ranking Demo")
st.markdown("**Analyze SEC 10-Q Filings - Amazon & Apple**")

# Get API key from secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- Sidebar ---
with st.sidebar:
    st.header("Step 1: Load SEC 10-Q Filings")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Amazon and Apple 10-Q PDFs", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload the provided SEC 10-Q PDF files for Amazon and Apple"
    )
    
    process_button = st.button("Process Documents", type="primary")
    
    # Display processed files
    if st.session_state.processed_files:
        st.success(f"✅ Processed files:")
        for file in st.session_state.processed_files:
            st.text(f"• {file}")
    
    st.divider()
    
    # Re-ranking control slider
    st.header("Step 4: Re-Ranking Control")
    k_slider = st.slider(
        "Max chunks to retrieve (k)", 
        min_value=1, 
        max_value=10, 
        value=4, 
        help="Adjust to explore how the number of retrieved chunks affects results"
    )
    
    st.divider()
    
    # Task instructions
    st.header("Task Workflow")
    st.markdown("""
    **Step 2:** Ask retrieval questions:
    - 'Summarize the financial performance this quarter'
    - 'What are the main risks?'
    - 'Explain the company's cash flow situation'
    
    **Step 3:** Compare across companies - ask the same questions for both
    
    **Step 4:** Adjust the k-value slider and observe changes
    """)

# --- Main Logic ---

# Process uploaded files
if process_button:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file.")
    elif not openai_api_key:
        st.error("❌ OpenAI API key not found. Please add it to your secrets.toml.")
    else:
        with st.spinner("Processing documents... (Extracting, Chunking, Embedding)"):
            
            # Extract text from PDFs
            raw_text = ""
            file_names = []
            for file in uploaded_files:
                file_names.append(file.name)
                reader = PdfReader(file)
                for page in reader.pages:
                    raw_text += page.extract_text() or ""
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len
            )
            text_chunks = text_splitter.split_text(raw_text)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            st.session_state.vectorstore = Chroma.from_texts(
                texts=text_chunks, 
                embedding=embeddings
            )
            
            # Store processed file names
            st.session_state.processed_files = file_names
            
            # Clear old conversation
            st.session_state.messages = []
            
        st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) with {len(text_chunks)} chunks. You can now ask questions!")

# Chat interface
st.header("Step 2 & 3: Ask Questions and Compare")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander(f"📚 Show Sources (k={message.get('k_value', 'N/A')})"):
                for source in message["sources"]:
                    st.json(source)

# Handle user input
if prompt := st.chat_input("Ask a question about the SEC filings..."):
    
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if RAG is ready
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process documents first.")
    elif not openai_api_key:
         st.error("❌ OpenAI API key not configured.")
    else:
        with st.spinner("Retrieving, Re-ranking, and Generating response..."):
            
            # Create RAG chain with current k value
            rag_chain, retriever = get_rag_chain(
                st.session_state.vectorstore,
                openai_api_key,
                k_slider
            )
            
            # Get response
            response = rag_chain.invoke(prompt)
            
            # Get source documents
            source_docs = retriever.get_relevant_documents(prompt)
            
            # Extract answer text
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            # Format sources for display
            sources_data = []
            for i, doc in enumerate(source_docs):
                sources_data.append({
                    f"Source (Ranked #{i+1})": {
                        "content_snippet": doc.page_content[:250] + "...",
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                    }
                })

            # Add assistant message to session
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources_data,
                "k_value": k_slider
            })
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander(f"📚 Show {len(sources_data)} Sources (k={k_slider})"):
                    for source in sources_data:
                        st.json(source)

# Footer with instructions
st.divider()
st.markdown("""
### 💡 Tips for Analysis:
- Upload both Amazon and Apple 10-Q filings for comparative analysis
- Try the same questions for each company to identify differences
- Experiment with the k-value slider to see how it affects retrieval quality
- Notice how different k values may surface different relevant information
""")

# Display current configuration status
with st.expander("🔧 Current Configuration"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vector Store", "✅ Ready" if st.session_state.vectorstore else "❌ Not Ready")
    with col2:
        st.metric("Current k-value", k_slider)
    with col3:
        st.metric("Files Processed", len(st.session_state.processed_files))
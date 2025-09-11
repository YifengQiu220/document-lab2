import streamlit as st
from openai import OpenAI
import tiktoken

# Page title
st.title("Lab 3 - Educational Chatbot for Kids 🎓")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    ("GPT-4o Mini", "GPT-4o")
)
model_to_use = "gpt-4o-mini" if model_option == "GPT-4o Mini" else "gpt-4o"

# Lab 3A: Basic streaming 
# Lab 3B: Message-based buffer
st.sidebar.markdown("### Lab 3B: Message Buffer")
buffer_size = st.sidebar.slider(
    "Keep last N exchanges",
    min_value=1,
    max_value=10,
    value=2,
    help="Number of user-assistant exchanges to keep"
)

# Lab 3B (Token Buffer part): Token-based buffer
st.sidebar.markdown("### Lab 3B: Token Buffer")
max_tokens = st.sidebar.slider(
    "Max tokens in context",
    min_value=100,
    max_value=4000,
    value=1000,
    step=100,
    help="Maximum tokens to send to the LLM"
)

buffer_mode = st.sidebar.radio(
    "Buffer Mode",
    ["Message-based", "Token-based"],
    help="Choose how to limit conversation context"
)

# Lab 3C: Educational mode for kids
st.sidebar.markdown("### Lab 3C: Educational Mode")
educational_mode = st.sidebar.checkbox(
    "Enable Kid-Friendly Mode (10 years old)",
    value=True,
    help="Simplifies language and adds interactive learning"
)

# Initialize OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

# Initialize tokenizer
@st.cache_resource
def get_tokenizer():
    """Get tokenizer for GPT-4o models"""
    return tiktoken.get_encoding("cl100k_base")

# Token counting function
def count_tokens(messages):
    """Count tokens in a list of messages"""
    encoding = get_tokenizer()
    total_tokens = 0
    
    for message in messages:
        total_tokens += len(encoding.encode(message["role"]))
        total_tokens += len(encoding.encode(message["content"]))
        total_tokens += 4
    
    total_tokens += 3
    return total_tokens

# Message-based buffer function
def get_message_buffered(messages, buffer_size):
    """Returns the last N exchanges"""
    if len(messages) <= 1:
        return messages
    
    # Keep system message if exists
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    
    messages_to_keep = buffer_size * 2
    if len(other_msgs) > messages_to_keep:
        other_msgs = other_msgs[-messages_to_keep:]
    
    return system_msgs + other_msgs

# Token-based buffer function
def get_token_buffered(messages, max_tokens):
    """Returns messages that fit within max_tokens limit"""
    if not messages:
        return []
    
    # Keep system message
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    
    result = []
    total_tokens = count_tokens(system_msgs)
    
    for message in reversed(other_msgs):
        message_tokens = count_tokens([message])
        if total_tokens + message_tokens <= max_tokens:
            result.insert(0, message)
            total_tokens += message_tokens
        else:
            break
    
    return system_msgs + result

# Initialize conversation state for Lab 3C
if "waiting_for_more_info" not in st.session_state:
    st.session_state.waiting_for_more_info = False

if "current_topic" not in st.session_state:
    st.session_state.current_topic = None

# Initialize message history
if "messages" not in st.session_state:
    if educational_mode:
        st.session_state.messages = [
            {"role": "system", "content": """You are a friendly educational assistant for 10-year-old children. 
            Follow these rules:
            1. Use simple, clear language that a 10-year-old can understand
            2. After answering a question, ALWAYS ask: "Do you want more info?"
            3. If they say yes, provide more details in a fun, engaging way
            4. If they say no, ask "What else can I help you with?"
            5. Use examples and comparisons kids can relate to
            6. Be encouraging and positive
            7. Keep answers concise initially, then expand if asked"""},
            {"role": "assistant", "content": "Hi! I'm your learning buddy! 🌟 What would you like to learn about today?"}
        ]
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

# Clear conversation button
if st.sidebar.button("Clear Conversation 🗑️"):
    if educational_mode:
        st.session_state.messages = [
            {"role": "system", "content": """You are a friendly educational assistant for 10-year-old children. 
            Follow these rules:
            1. Use simple, clear language that a 10-year-old can understand
            2. After answering a question, ALWAYS ask: "Do you want more info?"
            3. If they say yes, provide more details in a fun, engaging way
            4. If they say no, ask "What else can I help you with?"
            5. Use examples and comparisons kids can relate to
            6. Be encouraging and positive
            7. Keep answers concise initially, then expand if asked"""},
            {"role": "assistant", "content": "Hi! I'm your learning buddy! 🌟 What would you like to learn about today?"}
        ]
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
    st.session_state.waiting_for_more_info = False
    st.session_state.current_topic = None
    st.rerun()

# Get buffered messages
if buffer_mode == "Message-based":
    buffered_messages = get_message_buffered(st.session_state.messages, buffer_size)
else:
    buffered_messages = get_token_buffered(st.session_state.messages, max_tokens)

# Display metrics
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Messages", len(st.session_state.messages))
with col2:
    st.metric("Buffered Messages", len(buffered_messages))

# Display all messages (skip system messages)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Lab 3C: Special handling for "Do you want more info?" responses
def handle_lab3c_response(user_input):
    """Process user input in educational mode"""
    user_input_lower = user_input.lower().strip()
    
    # Check if answering "Do you want more info?"
    if st.session_state.waiting_for_more_info:
        if any(word in user_input_lower for word in ['yes', 'yeah', 'sure', 'ok', 'okay', 'yep', 'please']):
            prompt = f"Please provide more detailed, fun information about {st.session_state.current_topic} for a 10-year-old. Include interesting facts or examples."
            st.session_state.waiting_for_more_info = True
        elif any(word in user_input_lower for word in ['no', 'nope', 'nah', 'not']):
            prompt = "Great! What else would you like to learn about?"
            st.session_state.waiting_for_more_info = False
            st.session_state.current_topic = None
        else:
            # New question
            st.session_state.current_topic = user_input
            st.session_state.waiting_for_more_info = True
            prompt = user_input
    else:
        # New topic
        st.session_state.current_topic = user_input
        st.session_state.waiting_for_more_info = True
        prompt = user_input
    
    return prompt

# User input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Lab 3C: Process input if in educational mode
    if educational_mode:
        processed_prompt = handle_lab3c_response(prompt)
        # Add instruction to ask "Do you want more info?"
        if st.session_state.waiting_for_more_info and "what else" not in processed_prompt.lower():
            processed_prompt += "\n\nRemember to end your response by asking 'Do you want more info?'"
    else:
        processed_prompt = prompt
    
    # Get buffered messages
    if buffer_mode == "Message-based":
        messages_for_api = get_message_buffered(st.session_state.messages, buffer_size)
    else:
        messages_for_api = get_token_buffered(st.session_state.messages, max_tokens)
    
    # Get AI response
    with st.chat_message("assistant"):
        if educational_mode:
            st.caption("🎓 Kid-friendly mode activated")
        
        stream = st.session_state.client.chat.completions.create(
            model=model_to_use,
            messages=messages_for_api,
            stream=True,
            temperature=0.7
        )
        response = st.write_stream(stream)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display current mode info
st.markdown("---")
if educational_mode:
    st.info("🎓 **Lab 3C - Educational Mode Active**: Responses are simplified for 10-year-olds with interactive follow-ups")
    
    # Example for "what is baseball?"
    with st.expander("📚 Try this example: 'What is baseball?'"):
        st.write("""
        The bot will:
        1. Give a simple explanation of baseball
        2. Ask "Do you want more info?"
        3. If you say 'yes' → More details about rules, teams, etc.
        4. If you say 'no' → Ask what else you want to learn
        """)
else:
    st.caption(f"**Buffer Mode**: {buffer_mode} | Messages: {len(buffered_messages)}/{len(st.session_state.messages)}")
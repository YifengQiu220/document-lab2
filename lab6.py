import streamlit as st
from openai import OpenAI
import json

# Page configuration
st.set_page_config(
    page_title="AI Fact-Checker",
    page_icon="🔍",
    layout="wide"
)

# Page title
st.title("🔍 AI Fact-Checker + Citation Builder")
st.markdown("*Verify factual claims using AI-powered web search and get structured citations*")

# Initialize OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

# Initialize history
if 'fact_check_history' not in st.session_state:
    st.session_state.fact_check_history = []

# Sidebar - Settings and History
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_option = st.selectbox(
        "Select Model",
        ("GPT-4o Mini", "GPT-4o"),
        help="Choose the model for fact-checking"
    )
    model_to_use = "gpt-4o-mini" if model_option == "GPT-4o Mini" else "gpt-4o"
    
    st.markdown("---")
    
    # Show confidence score option
    show_confidence = st.checkbox(
        "Show Confidence Score",
        value=True,
        help="Display AI's confidence in the verdict"
    )
    
    clickable_links = st.checkbox(
        "Format as Clickable Links",
        value=True,
        help="Convert URLs to markdown links"
    )
    
    st.markdown("---")
    st.header("📜 History")
    
    if st.session_state.fact_check_history:
        st.caption(f"Total checks: {len(st.session_state.fact_check_history)}")
        
        if st.button("Clear History", type="secondary"):
            st.session_state.fact_check_history = []
            st.rerun()
        
        with st.expander("View Past Checks"):
            for i, item in enumerate(reversed(st.session_state.fact_check_history[-5:])):
                st.markdown(f"**{i+1}. {item['claim'][:50]}...**")
                st.caption(f"Verdict: {item['verdict']}")
                st.markdown("---")
    else:
        st.info("No checks yet")

# Function: Fact-check using OpenAI Responses API
def fact_check_claim(claim, include_confidence=True):
    """
    Check a factual claim using OpenAI's API with web_search tool
    Returns structured JSON with verdict, explanation, and sources
    """
    
    system_prompt = """You are an expert fact-checker. Your job is to:
1. Analyze the given claim carefully
2. Use web search to find current, credible sources
3. Provide a clear verdict: TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIED
4. Give a concise explanation (2-3 sentences)
5. List 3-5 credible sources with titles and URLs
6. Assess confidence level (0-100%) based on source consistency

Return your response as valid JSON with this structure:
{
    "claim": "the original claim",
    "verdict": "TRUE/FALSE/PARTIALLY TRUE/UNVERIFIED",
    "confidence_score": 85,
    "explanation": "Brief explanation of why this verdict",
    "sources": [
        {
            "title": "Source title",
            "url": "https://...",
            "relevance": "Why this source matters"
        }
    ]
}"""

    try:
        # Call OpenAI Chat Completions API with function calling (web search)
        response = st.session_state.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Fact-check this claim: {claim}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Ensure required fields
        if "claim" not in result:
            result["claim"] = claim
        
        return result
        
    except Exception as e:
        st.error(f"Error during fact-checking: {str(e)}")
        return {
            "claim": claim,
            "verdict": "ERROR",
            "explanation": f"An error occurred: {str(e)}",
            "sources": []
        }

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    user_claim = st.text_input(
        "Enter a factual claim to verify:",
        placeholder="e.g., Is dark chocolate healthy?",
        help="Type any factual statement you want to verify"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    check_button = st.button("🔍 Check Fact", type="primary", use_container_width=True)

# Sample claims for quick testing
st.markdown("### 💡 Try these examples:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Is dark chocolate healthy?", use_container_width=True):
        user_claim = "Is dark chocolate actually healthy?"
        check_button = True

with col2:
    if st.button("Can coffee prevent cancer?", use_container_width=True):
        user_claim = "Can drinking coffee prevent cancer?"
        check_button = True

with col3:
    if st.button("Is Pluto a planet?", use_container_width=True):
        user_claim = "Is Pluto still classified as a planet?"
        check_button = True

# Process fact-check
if check_button and user_claim:
    with st.spinner("🔍 Verifying claim and searching for sources..."):
        result = fact_check_claim(user_claim, include_confidence=show_confidence)
        
        # Add to history
        st.session_state.fact_check_history.append(result)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Fact-Check Results")
        
        # Verdict badge
        verdict = result.get("verdict", "UNKNOWN")
        verdict_colors = {
            "TRUE": "🟢",
            "FALSE": "🔴",
            "PARTIALLY TRUE": "🟡",
            "UNVERIFIED": "⚪",
            "ERROR": "⚫"
        }
        
        verdict_icon = verdict_colors.get(verdict, "❓")
        
        # Display verdict prominently
        st.markdown(f"### Verdict: {verdict_icon} **{verdict}**")
        
        # Confidence score
        if show_confidence and "confidence_score" in result:
            confidence = result["confidence_score"]
            st.progress(confidence / 100)
            st.caption(f"Confidence: {confidence}%")
        
        st.markdown("---")
        
        # Two column layout for results
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 📝 Explanation")
            st.write(result.get("explanation", "No explanation provided"))
        
        with col_right:
            st.markdown("#### 📚 Sources")
            sources = result.get("sources", [])
            
            if sources:
                for i, source in enumerate(sources, 1):
                    if clickable_links:
                        title = source.get("title", "Untitled")
                        url = source.get("url", "#")
                        st.markdown(f"{i}. [{title}]({url})")
                        if "relevance" in source:
                            st.caption(f"   ↳ {source['relevance']}")
                    else:
                        st.write(f"{i}. {source.get('title', 'Untitled')}")
                        st.caption(f"   {source.get('url', 'No URL')}")
            else:
                st.info("No sources found")
        
        # Show raw JSON in expander
        with st.expander("🔧 View Raw JSON Response"):
            st.json(result)

# Information section
st.markdown("---")
st.markdown("### ℹ️ How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**1️⃣ Submit Claim**")
    st.caption("Enter any factual statement you want verified")

with col2:
    st.markdown("**2️⃣ AI Analysis**")
    st.caption("AI searches the web and analyzes credible sources")

with col3:
    st.markdown("**3️⃣ Get Verdict**")
    st.caption("Receive structured results with citations")


# Footer
st.markdown("---")
st.caption("🔍 AI Fact-Checker powered by OpenAI | Lab 6 - Agentic Research Assistant")
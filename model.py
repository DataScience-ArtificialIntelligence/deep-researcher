import streamlit as st
import requests
import json
import os
from datetime import datetime
import time
from dotenv import load_dotenv
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables silently
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="DeepResearch Assistant",
    page_icon="🔍",
    layout="wide"
)

# Hide specific service information
SERP_API_KEY = os.getenv("serp_api_key", "39de6ee154a641a07eab3f85efb116fd7eee2983bd6ba45b3c9a1e8c874c9490")

# Model paths
LOCAL_MODEL_PATH = "./models/llama-3.2-1B"

# Initialize session state variables if they don't exist
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'search_status' not in st.session_state:
    st.session_state.search_status = None
if 'summary_status' not in st.session_state:
    st.session_state.summary_status = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'pipe' not in st.session_state:
    st.session_state.pipe = None

# Function to load model (only once)
def load_model():
    if not st.session_state.model_loaded:
        with st.spinner("Loading Llama 3.2 model... This may take a minute..."):
            try:
                # Load tokenizer and model from local path
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
                st.session_state.model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_MODEL_PATH,
                    device_map="auto"  # Use GPU if available
                )
                
                # Create the pipeline
                st.session_state.pipe = pipeline(
                    "text-generation",
                    model=st.session_state.model,
                    tokenizer=st.session_state.tokenizer
                )
                
                st.session_state.model_loaded = True
                return True
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return False
    return True

# Function to perform web search
def search_web(query, num_results=5):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERP_API_KEY,
            "num": num_results,
            "tbm": "isch"  # This parameter requests image results
        }
        
        # First get regular search results
        regular_params = {
            "engine": "google",
            "q": query,
            "api_key": SERP_API_KEY,
            "num": num_results
        }
        
        regular_response = requests.get("https://serpapi.com/search", params=regular_params)
        regular_results = regular_response.json()
        
        # Then get image results
        image_response = requests.get("https://serpapi.com/search", params=params)
        image_results = image_response.json()
        
        # Extract relevant information from search results
        search_results = []
        
        if "organic_results" in regular_results:
            for result in regular_results["organic_results"]:
                search_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "image": None  # Will be populated with images if available
                })
        
        # Add images to results if available
        if "images_results" in image_results and len(image_results["images_results"]) > 0:
            # Add up to 5 images to the beginning of results
            image_count = min(5, len(image_results["images_results"]))
            for i in range(image_count):
                if i < len(image_results["images_results"]):
                    img_data = image_results["images_results"][i]
                    if i < len(search_results):
                        search_results[i]["image"] = img_data.get("thumbnail", None)
                    else:
                        # If we have more images than text results, add as image-only results
                        search_results.append({
                            "title": img_data.get("title", "Image Result"),
                            "link": img_data.get("original", ""),
                            "snippet": img_data.get("snippet", ""),
                            "image": img_data.get("thumbnail", None)
                        })
        
        return search_results
    except Exception as e:
        st.error(f"Error searching for information: {str(e)}")
        return []

# Function to generate summary using local Llama model
def generate_summary(search_results, query):
    try:
        # Make sure model is loaded
        if not st.session_state.model_loaded:
            if not load_model():
                return "Failed to load the model. Please check your model installation."
        
        # Prepare the context from search results
        context = "\n\n".join([
            f"Source: {result['title']}\nURL: {result['link']}\nSummary: {result['snippet']}"
            for result in search_results
        ])
        
        # Create the prompt for the local Llama model
        prompt = f"""You are a research assistant tasked with summarizing information from web search results.
        
RESEARCH QUERY: {query}

SEARCH RESULTS:
{context}

Based on these search results, provide a comprehensive research summary that:
1. Synthesizes the key information relevant to the query
2. Identifies main findings, perspectives, and any consensus or disagreements
3. Cites specific sources when presenting information
4. Notes any limitations in the available information
5. Suggests areas for further research if needed

FORMAT YOUR RESPONSE AS FOLLOWS:
## Research Summary
[Your main summary here - approximately 3-4 paragraphs]

## Key Findings
- [Bullet points of the most important discoveries]

## Sources Overview
- [Brief assessment of the quality and relevance of sources]

## Further Research Directions
- [Suggestions for additional areas to explore]
"""
        
        # Generate response using local pipeline
        messages = [{"role": "user", "content": prompt}]
        
        # Get response from the model
        response = st.session_state.pipe(
            messages,
            max_length=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract response content - format may vary depending on the pipeline output
        # This extracts the assistant's response from the generated text
        response_text = response[0]["generated_text"]
        
        # Extract just the assistant's reply part (format depends on model output)
        # For Llama-3.2-1B-Instruct, we need to extract the assistant's response
        if "<assistant>" in response_text:
            assistant_response = response_text.split("<assistant>")[1].split("</assistant>")[0].strip()
        else:
            # If we can't find the marker, just return everything after the prompt
            assistant_response = response_text.replace(prompt, "").strip()
        
        return assistant_response
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return f"Failed to generate summary due to an error: {str(e)}"

# Custom CSS to inject
custom_css = """
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    .search-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .result-header {
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .animate-pulse {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .footer {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
    }
    
    .result-image {
        max-height: 150px;
        max-width: 100%;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .result-item {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #ffffff;
    }
    
    .model-info {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 4px solid #3a7bd5;
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Main header with custom styling
st.markdown('<h1 class="main-header">🔍 DeepResearch Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
Enter any research topic or question, and I'll analyze information from various sources 
to provide you with a comprehensive research summary.
""")

# Display model information
st.markdown('<div class="model-info">', unsafe_allow_html=True)
st.markdown("**Model:** Llama 3.2 1B (Local Inference)")
st.markdown("Using locally downloaded model for inference - no API calls required")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model settings
    st.subheader("Model Settings")
    
    # Load model button
    if not st.session_state.model_loaded:
        if st.button("Load Llama 3.2 Model"):
            load_model()
    else:
        st.success("✅ Llama 3.2 Model loaded!")
    
    # Search settings
    st.subheader("Search Settings")
    num_results = st.slider("Number of search results", min_value=3, max_value=15, value=7)
    
    # Search history
    st.subheader("📚 Search History")
    if st.session_state.search_history:
        for i, (timestamp, query) in enumerate(st.session_state.search_history):
            if st.button(f"{timestamp}: {query[:30]}...", key=f"history_{i}"):
                st.session_state.search_query = query
                st.experimental_rerun()
    else:
        st.write("No search history yet")
    
    # Information
    st.markdown("---")
    st.markdown("""
    ### About
    DeepResearch Assistant helps you quickly research topics by searching multiple sources and generating a comprehensive summary using Llama 3.2 locally.
    
    ### Features
    - Intelligent web search
    - Local Llama 3.2 inference
    - Advanced analysis
    - Citation of sources
    - Future research suggestions
    """)

# Search input in a styled container
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_query = st.text_input("Research Topic or Question", 
                         value=st.session_state.get('search_query', ''),
                         key="search_query",
                         placeholder="Enter your research topic here...")

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("🔍 Research", use_container_width=True)
with col2:
    clear_button = st.button("🧹 Clear Results", use_container_width=True)
with col3:
    # Model loading status
    if st.session_state.model_loaded:
        st.success("Model ready for inference", icon="✅")
    else:
        st.warning("Model not loaded - click 'Load Llama 3.2 Model' in sidebar", icon="⚠️")
st.markdown('</div>', unsafe_allow_html=True)

# Status containers for animations
search_status_container = st.empty()
summary_status_container = st.empty()
results_container = st.container()

# Animation frames for more visual interest
research_animations = [
    "📚 Exploring research databases...",
    "🔎 Scanning scholarly articles...",
    "📊 Analyzing data patterns...",
    "📝 Gathering recent publications...",
    "🧩 Connecting related concepts...",
    "📖 Reviewing academic sources...",
    "🔬 Investigating key findings..."
]

summary_animations = [
    "🧠 Processing information...",
    "📊 Analyzing data relationships...",
    "🔄 Synthesizing concepts...",
    "📋 Organizing key findings...",
    "💡 Identifying insights...",
    "📝 Drafting conclusions...",
    "✏️ Refining summary..."
]

# Process search when button is clicked
if search_button and search_query:
    # First check if model is loaded
    if not st.session_state.model_loaded:
        if st.button("Load Model Now"):
            load_model()
        search_status_container.warning("⚠️ Please load the Llama 3.2 model from the sidebar first")
    else:
        # Update search history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.search_history.append((timestamp, search_query))
        
        # Animated web search progress
        search_status_container.info("🔎 Initiating comprehensive research...")
        progress_bar_search = search_status_container.progress(0)
        
        # Simulate search progress with animations
        for i in range(101):
            time.sleep(0.01)  # Adjust for desired animation speed
            progress_bar_search.progress(i)
            if i % 15 == 0:
                search_status_container.info(random.choice(research_animations))
        
        # Perform the actual search
        search_results = search_web(search_query, num_results)
        
        if search_results:
            search_status_container.success("✅ Research data collected!")
            
            # Animated summary generation progress
            summary_status_container.info("🧠 Analyzing research data with Llama 3.2...")
            progress_bar_summary = summary_status_container.progress(0)
            
            # Simulate summary progress with animations
            for i in range(101):
                time.sleep(0.03)  # Adjust for desired animation speed
                progress_bar_summary.progress(i)
                if i % 14 == 0:
                    summary_status_container.info(random.choice(summary_animations))
            
            # Generate the actual summary with local model
            summary = generate_summary(search_results, search_query)
            summary_status_container.success("✅ Research summary generated with Llama 3.2!")
            
            # Store results
            st.session_state.current_results = {
                "query": search_query,
                "search_results": search_results,
                "summary": summary,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            search_status_container.error("❌ No research data found. Please try a different query.")
            summary_status_container.empty()

# Clear results if clear button is clicked
if clear_button:
    st.session_state.current_results = None
    search_status_container.empty()
    summary_status_container.empty()

# Display results if they exist
with results_container:
    if st.session_state.current_results:
        # Results header with custom styling
        st.markdown(f'<div class="result-header"><h2>Research Results: {st.session_state.current_results["query"]}</h2></div>', unsafe_allow_html=True)
        st.caption(f"Generated on: {st.session_state.current_results['timestamp']} with Llama 3.2 1B (Local)")
        
        # Create tabs for summary and raw search results
        tab1, tab2 = st.tabs(["Research Summary", "Source Information"])
        
        with tab1:
            st.markdown(st.session_state.current_results["summary"])
            
            # Add a download button for the summary
            summary_text = st.session_state.current_results["summary"]
            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with tab2:
            # Display search results with images if available
            for i, result in enumerate(st.session_state.current_results["search_results"]):
                st.markdown(f'<div class="result-item">', unsafe_allow_html=True)
                
                # Display image if available
                if result["image"]:
                    st.image(result["image"], width=200, caption=result["title"][:50])
                
                st.markdown(f"### {i+1}. {result['title']}")
                st.write(result['snippet'])
                st.write(f"[Visit Source]({result['link']})")
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("DeepResearch Assistant - Powered by Llama 3.2 (Local Inference)")
st.markdown("© 2025 DeepResearch - All Rights Reserved")
st.markdown('</div>', unsafe_allow_html=True)

# Hide Streamlit style and menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
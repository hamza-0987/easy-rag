import streamlit as st
import os
import sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easy_rag_pipeline import load_config, simple_rag_pipeline

st.set_page_config(page_title="Easy RAG Pipeline Demo", layout="wide")

st.title("📄 Easy RAG Pipeline Demo")

# --- Instructions and Setup ---
st.sidebar.header("Setup")
st.sidebar.info(
    "This is a demo of the Easy RAG Pipeline. "
    "You can ask questions about a document, and the pipeline will retrieve relevant information "
    "and generate an answer."
)

# For a real app, you might let the user upload a file
# For this demo, we'll use the sample document.
source_document = os.path.join(os.path.dirname(__file__), 'sample_document.txt')
source_type = 'txt'

# --- Main App ---
st.header("Ask a Question")

query = st.text_input("Enter your question about the document:", "What is FAISS used for in a RAG pipeline?")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Load configuration
                config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
                config = load_config(config_path)

                # Check for API key
                if not config.get('api_keys', {}).get('openai'):
                    st.error("OpenAI API key not found. Please set it in your .env file in the project root.")
                else:
                    # Run the pipeline
                    answer = simple_rag_pipeline(
                        query=query,
                        source_path=source_document,
                        source_type=source_type,
                        config=config
                    )
                    st.success(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.sidebar.header("Document Content")
with open(source_document, 'r') as f:
    st.sidebar.text_area("Sample Document", f.read(), height=300)

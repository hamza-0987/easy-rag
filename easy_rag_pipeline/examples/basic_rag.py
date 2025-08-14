import os
import sys
from easy_rag_pipeline import load_config, simple_rag_pipeline, setup_logging

# Add the root directory to the Python path to allow importing the library
# This is for demonstration purposes when running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    A basic example demonstrating the use of the simple_rag_pipeline.

    Usage:
    1. Make sure you have an API key set in a .env file in the root directory.
    2. Run this script from the project root directory:
       python examples/basic_rag.py
    """
    # Setup logging
    logger = setup_logging()

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Check for API key
    if not config.get('llm', {}).get('api_key'):
        provider = config.get('llm', {}).get('provider', 'the selected provider')
        key_env_var = f"{provider.upper()}_API_KEY"
        if provider == 'gemini':
            key_env_var = 'GOOGLE_API_KEY'
        logger.error(f"API key for provider '{provider}' not found. Please set {key_env_var} in your .env file.")
        return

    # Define the source document and the query
    source_document = os.path.join(os.path.dirname(__file__), 'sample_document.txt')
    query = "What is Retrieval-Augmented Generation (RAG)?"

    logger.info(f"Querying the RAG pipeline with: '{query}'")
    logger.info(f"Source document: '{source_document}'")

    # Run the simple, all-in-one pipeline
    answer = simple_rag_pipeline(
        query=query,
        source_path=source_document,
        source_type='txt',
        config=config
    )

    print("\n" + "="*30)
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    print("="*30)


if __name__ == "__main__":
    main()

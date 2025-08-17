import os
from easy_rag_pipeline import load_config, multimodal_rag_pipeline, setup_logging

def main():
    """
    An example demonstrating the use of the multimodal_rag_pipeline.

    Usage:
    1. Make sure you have an OpenAI API key set in a .env file.
    2. Enable multimodal mode in examples/config.yaml.
    3. Run this script from the project root directory:
       python examples/multimodal_rag.py
    """
    # Setup logging
    logger = setup_logging()

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)

    if not config.get("multimodal", {}).get("enabled", False):
        logger.warning("Multimodal mode is not enabled in config.yaml. Skipping example.")
        print("\nNOTE: Please enable multimodal mode in examples/config.yaml to run this demo.")
        return

    logger.info("Configuration loaded successfully.")

    # Check for API key
    if not config.get('llm', {}).get('api_key'):
        provider = config.get('llm', {}).get('provider', 'the selected provider')
        key_env_var = f"{provider.upper()}_API_KEY"
        if provider == 'gemini':
            key_env_var = 'GOOGLE_API_KEY'
        logger.error(f"API key for provider '{provider}' not found. Please set {key_env_var} in your .env file.")
        return

    # Define the source documents and the query
    text_document = os.path.join(os.path.dirname(__file__), 'sample_document.txt')
    image_directory = os.path.join(os.path.dirname(__file__), '.')
    query = "Based on the text and the image provided, what is this demo about?"

    logger.info(f"Querying the Multimodal RAG pipeline with: '{query}'")
    logger.info(f"Text source: '{text_document}'")
    logger.info(f"Image source directory: '{image_directory}'")

    # Run the multimodal pipeline
    answer = multimodal_rag_pipeline(
        query=query,
        text_source_path=text_document,
        image_directory_path=image_directory,
        config=config
    )

    print("\n" + "="*30)
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    print("="*30)


if __name__ == "__main__":
    # Add project root to path to allow direct execution
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()

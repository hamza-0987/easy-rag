from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function(embedding_config: dict):
    """
    Creates and returns an embedding function based on the provided configuration.

    Args:
        embedding_config (dict): A dictionary containing embedding model details.
            Example for OpenAI:
            {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "api_key": "..."
            }
            Example for HuggingFace:
            {
                "provider": "huggingface",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }

    Returns:
        An embedding function object from LangChain.
    """
    provider = embedding_config.get("provider", "openai").lower()

    if provider == "openai":
        return OpenAIEmbeddings(
            model=embedding_config.get("model", "text-embedding-ada-002"),
            openai_api_key=embedding_config.get("api_key")
        )
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

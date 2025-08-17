from .ingest import chunk_documents, load_pdf, load_website, load_text, load_images_from_directory
from .embed import get_embedding_function
from .store import create_vector_store, create_multi_vector_retriever
from .retrieve import retrieve_documents
from .generate import generate_answer, generate_multimodal_answer
import os

# A more practical approach is to separate indexing from querying.
# 1. Create a vector store once.
# 2. Query it multiple times.

def create_and_persist_vector_store(source_path: str, source_type: str, config: dict, save_path: str = "vector_store"):
    """
    Loads data, chunks it, creates embeddings, and saves a vector store.
    This is the "indexing" part of the pipeline.

    Args:
        source_path (str): Path to the data (file path or URL).
        source_type (str): Type of data ('pdf', 'website', 'txt').
        config (dict): The main configuration dictionary.
        save_path (str): Directory to save the FAISS index.
    """
    print("Loading documents...")
    if source_type == 'pdf':
        docs = load_pdf(source_path)
    elif source_type == 'website':
        docs = load_website(source_path)
    elif source_type == 'txt':
        docs = load_text(source_path)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    print("Chunking documents...")
    chunks = chunk_documents(docs, **config.get('chunking', {}))

    print("Creating embedding function...")
    embedding_function = get_embedding_function(config['embedding'])

    print("Creating and persisting vector store...")
    vector_store = create_vector_store(chunks, embedding_function, config['vector_store'])

    # Save the vector store if it's a type that supports it (like FAISS)
    if config['vector_store']['provider'] == 'faiss':
        vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")

    return vector_store


def query_rag_pipeline(query: str, vector_store, config: dict):
    """
    Queries the RAG pipeline using a pre-existing vector store.
    This is the "querying" part of the pipeline.

    Args:
        query (str): The user's query.
        vector_store: The loaded vector store object.
        config (dict): The main configuration dictionary.

    Returns:
        str: The generated answer.
    """
    print("Retrieving documents...")
    retrieved_docs = retrieve_documents(query, vector_store, **config.get('retrieval', {}))

    print("Generating answer...")
    answer = generate_answer(query, retrieved_docs, config['llm'])

    return answer


def multimodal_rag_pipeline(query: str, text_source_path: str, image_directory_path: str, config: dict):
    """
    An all-in-one multimodal RAG pipeline.

    Args:
        query (str): The user's query.
        text_source_path (str): Path to the text document.
        image_directory_path (str): Path to the directory of images.
        config (dict): The main configuration dictionary.

    Returns:
        str: The generated answer.
    """
    if not config.get("multimodal", {}).get("enabled", False):
        raise ValueError("Multimodal mode is not enabled in the configuration.")

    print("Executing Multimodal RAG Pipeline...")

    # 1. Load text and image documents
    print("1. Loading documents...")
    text_docs = load_text(text_source_path)
    image_docs = load_images_from_directory(image_directory_path)

    # 2. Chunk text documents
    print("2. Chunking text documents...")
    text_chunks = chunk_documents(text_docs, **config.get('chunking', {}))

    # 3. Create embedding functions
    # For this example, we'll use the same text embedding model for images,
    # as specified in the config. A true multimodal setup might use a dedicated
    # image embedding model.
    print("3. Creating embedding functions...")
    text_embedding_fn = get_embedding_function(config['embedding'])
    image_embedding_fn = get_embedding_function(config['embedding']) # Placeholder

    # 4. Create Multi-Vector Retriever
    print("4. Creating Multi-Vector Retriever...")
    retriever = create_multi_vector_retriever(
        text_docs=text_chunks,
        image_docs=image_docs,
        text_embedding_fn=text_embedding_fn,
        image_embedding_fn=image_embedding_fn,
        llm_config=config['llm'],
        store_config=config['vector_store']
    )

    # 5. Retrieve documents
    print("5. Retrieving relevant documents...")
    retrieved_docs = retriever.invoke(query)

    # 6. Generate multimodal answer
    print("6. Generating multimodal answer...")
    answer = generate_multimodal_answer(query, retrieved_docs, config['llm'])

    return answer


# The simple, all-in-one pipeline as requested by the user for quick demos.
def simple_rag_pipeline(query: str, source_path: str, source_type: str, config: dict):
    """
    A simple, all-in-one RAG pipeline that performs all steps in a single call.
    This is less efficient as it re-indexes the data on every query.

    Args:
        query (str): The user's query.
        source_path (str): Path to the data (file path or URL).
        source_type (str): Type of data ('pdf', 'website', 'txt').
        config (dict): The main configuration dictionary.

    Returns:
        str: The generated answer.
    """
    print("Executing simple RAG pipeline (note: this re-indexes data on every call)...")

    print("1. Loading documents...")
    if source_type == 'pdf':
        docs = load_pdf(source_path)
    elif source_type == 'website':
        docs = load_website(source_path)
    elif source_type == 'txt':
        docs = load_text(source_path)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    print("2. Chunking documents...")
    chunks = chunk_documents(docs, **config.get('chunking', {}))

    print("3. Creating embedding function...")
    embedding_function = get_embedding_function(config['embedding'])

    print("4. Creating in-memory vector store...")
    vector_store = create_vector_store(chunks, embedding_function, config['vector_store'])

    print("5. Retrieving documents...")
    retrieved_docs = retrieve_documents(query, vector_store, **config.get('retrieval', {}))

    print("6. Generating answer...")
    answer = generate_answer(query, retrieved_docs, config['llm'])

    return answer

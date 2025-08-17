import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from PIL import Image
import io
import base64

# To be added in the future:
# from langchain_community.vectorstores import Chroma, Pinecone, Weaviate

def create_vector_store(chunks: list, embedding_function, store_config: dict):
    """
    Creates a vector store from document chunks and an embedding function.

    Args:
        chunks (list): A list of document chunks.
        embedding_function: The embedding function to use.
        store_config (dict): Configuration for the vector store.
            Example:
            {
                "provider": "faiss"
            }

    Returns:
        A LangChain vector store object.
    """
    provider = store_config.get("provider", "faiss").lower()

    if provider == "faiss":
        # FAISS.from_documents creates the vector store in memory.
        # For persistence, you would use db.save_local(...) and FAISS.load_local(...)
        vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_function)
        return vector_db
    # Example for Chroma (requires `pip install chromadb`)
    # elif provider == "chroma":
    #     return Chroma.from_documents(documents=chunks, embedding=embedding_function)
    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")


def resize_base64_image(base64_string, size=(128, 128)):
    """Resizes a base64 encoded image."""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size)

    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_summary(base64_image, llm_config):
    """Generates a text summary for a base64 encoded image."""
    llm = ChatOpenAI(
        model=llm_config.get("image_summarize_model", "gpt-4o-mini"),
        api_key=llm_config.get("api_key")
    )

    chat_message = [
        {
            "type": "text",
            "text": "Describe this image in detail. What is the main subject, what is happening, and what context can be inferred? Be concise.",
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}",
        },
    ]

    return llm.invoke(chat_message).content


def create_multi_vector_retriever(text_docs, image_docs, text_embedding_fn, image_embedding_fn, llm_config, store_config):
    """
    Creates a multi-vector retriever for multimodal RAG.
    """
    vectorstore = create_vector_store([], text_embedding_fn, store_config)
    docstore = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=docstore, id_key="doc_id")

    # Add text documents
    doc_ids = [str(uuid.uuid4()) for _ in text_docs]
    summary_texts = [Document(page_content=doc.page_content, metadata={"doc_id": doc_ids[i]}) for i, doc in enumerate(text_docs)]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, text_docs)))

    # Add image documents
    if image_docs:
        # Get image summaries
        image_summaries = [get_image_summary(doc.page_content, llm_config) for doc in image_docs]

        # Add summaries to vector store
        img_doc_ids = [str(uuid.uuid4()) for _ in image_docs]
        summary_img = [Document(page_content=summary, metadata={"doc_id": img_doc_ids[i]}) for i, summary in enumerate(image_summaries)]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_doc_ids, image_docs)))

        # Add raw images to vector store
        # To-do: Add a proper image embedding model here
        # For now, we are just using the text embedding model on the summary
        # which is not ideal, but works as a placeholder.

    return retriever

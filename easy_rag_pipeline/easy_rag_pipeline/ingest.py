import base64
import mimetypes
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_core.documents import Document

def load_pdf(file_path: str):
    """Loads a PDF file and returns a list of documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_website(url: str):
    """Loads content from a website and returns a list of documents."""
    loader = WebBaseLoader(url)
    return loader.load()

def load_text(file_path: str):
    """Loads a text file and returns a list of documents."""
    loader = TextLoader(file_path)
    return loader.load()

def chunk_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Chunks a list of documents into smaller pieces.

    Args:
        docs (list): A list of documents to be chunked.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def is_image_file(file_path: str) -> bool:
    """Checks if a file is an image based on its MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('image/')

def image_to_base64(file_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_images_from_directory(directory_path: str) -> list[Document]:
    """
    Loads all images from a directory, encodes them in base64, and returns them
    as a list of Document objects.

    Args:
        directory_path (str): The path to the directory containing images.

    Returns:
        list[Document]: A list of Document objects, where the page_content is
                        the base64 encoded image.
    """
    image_documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and is_image_file(file_path):
            base64_image = image_to_base64(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            doc = Document(
                page_content=base64_image,
                metadata={
                    "source": file_path,
                    "type": "image",
                    "mime_type": mime_type
                }
            )
            image_documents.append(doc)
    return image_documents

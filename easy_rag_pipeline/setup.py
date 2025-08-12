from setuptools import setup, find_packages

setup(
    name="easy_rag_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langchain-community",
        "langchain-groq",
        "langchain-google-genai",
        "faiss-cpu",
        "pypdf",
        "tiktoken",
        "pyyaml",
        "python-dotenv",
        "streamlit",
    ],
    author="Jules",
    author_email="jules@example.com",
    description="A reusable and configurable RAG (Retrieval-Augmented Generation) pipeline.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/example/easy_rag_pipeline",
)

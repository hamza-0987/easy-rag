<div align="center">

```
 _____                         ____       _      ____  _                    _
| ____|_ __ ___  ___ ___ ____ / ___|  ___| | __ / ___|(_)_ __   ___   _ __ | | __
|  _| | '__/ _ \/ __/ __|_  / \___ \ / __| |/ / \___ \| | '_ \ / _ \ | '_ \| |/ /
| |___| | |  __/\__ \__ \/ /   ___) | (__|   <   ___) | | | | |  __/_| |_) |   <
|_____|_|  \___||___/___/___| |____/ \___|_|\_\ |____/|_|_| |_|\___(_) .__/|_|\_\
                                                                   |_|
```

**A flexible, configurable, and easy-to-use Retrieval-Augmented Generation (RAG) pipeline in a box.**

</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/easy-rag-pipeline.svg)](https://pypi.org/project/easy-rag-pipeline/)
[![Build Status](https://img.shields.io/travis/com/your-username/easy_rag_pipeline.svg)](https://travis-ci.com/your-username/easy_rag_pipeline)

</div>

---

**Easy RAG Pipeline** is a production-grade framework for building and deploying RAG applications. It handles the entire workflow from document ingestion to answer generation, allowing you to plug in different LLMs, embedding models, and vector stores with a simple, configuration-driven setup.

## ✨ Key Features

- **🔌 Pluggable Architecture**: Easily switch between LLMs (**OpenAI, Groq, Gemini, OpenRouter**), embedding models (**OpenAI, HuggingFace**), and vector stores (**FAISS**).
- **⚙️ Configuration-Driven**: No hard-coding. Manage all your settings—from chunk sizes to model names—through a single `config.yaml` file.
- **🚀 Efficient & Practical**: Includes two pipeline modes: a simple all-in-one for quick demos, and an advanced two-step process (index then query) for production efficiency.
- **🖼️ Multimodal Ready**: Process and reason over both text and images. The pipeline can ingest images, create text summaries of them, and use both in its context.
- **📚 Multi-Format Ingestion**: Out-of-the-box support for loading documents from PDFs, text files, websites, and now images.
- **📦 Ready-to-Use**: Comes with a CLI example, a Streamlit demo, and a multimodal example to get you started quickly.
- **🔧 Extensible by Design**: Clean, modular code that's easy to extend with your own custom components.

## 🏗️ Architecture

The pipeline uses a multi-vector retriever strategy for handling multimodal data. Text is chunked, while images are summarized. Both the raw text and the image summaries are embedded and used for retrieval.

```
[Text Docs]--+      +--[Image Docs]
     |       |      |       |
     v       |      v       v
[Chunk Text] | [Summarize Image]  <-- Vision LLM
     |       |      |
     |       +------+
     |              |
     v              v
[Embed Chunks & Summaries]
     |
     v
[Vector Store (FAISS)]<--+
     |                    |
[User Query] -> [Retrieve Docs]
     |
     v
[Retrieved Text & Images]
     |
     v
[Generate Answer]      <-- (Vision LLM: GPT-4o, Gemini)
     |
     v
   [Answer]
```

## 🚀 Getting Started

### 1. Installation

First, clone the repository and navigate into the project directory.

```bash
git clone https://github.com/your-username/easy_rag_pipeline.git
cd easy_rag_pipeline
```

Next, create and activate a virtual environment (recommended).

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install all required dependencies.

```bash
pip install -r requirements.txt
```

Finally, install the package in editable mode, which allows you to modify the code and see changes instantly.

```bash
pip install -e .
```

### 2. Configuration

This project is managed by a central configuration file and environment variables for your secrets.

**a. Set up API Keys:**

Copy the example environment file and add your secret API keys. The pipeline will automatically load the correct key based on the provider you choose in `config.yaml`.

```bash
cp .env.example .env
```
Now, edit `.env` and add your keys:
```
# For OpenAI
OPENAI_API_KEY="sk-..."

# For Groq
GROQ_API_KEY="gsk_..."

# For Google Gemini
GOOGLE_API_KEY="AIzaSy..."

# For OpenRouter
OPENROUTER_API_KEY="sk-or-..."
```

**b. Select your Components:**

Open `examples/config.yaml` to select the models and components you want to use. For example, to switch to Groq's Llama3 model:

```yaml
# examples/config.yaml

llm:
  provider: "groq"
  model: "llama3-8b-8192"
  temperature: 0.7
```

### 3. Run the Demo

You can test the pipeline with either the basic CLI example or the interactive Streamlit app.

**a. Basic CLI Example:**

This script will run the simple, all-in-one pipeline on the included sample document.

```bash
python examples/basic_rag.py
```

**b. Interactive Streamlit Demo:**

For a more visual experience, launch the Streamlit app.

```bash
streamlit run examples/streamlit_demo.py
```
This will open the demo in your web browser.

**c. Multimodal RAG Example:**

To test the multimodal capabilities, first enable it in `examples/config.yaml`:
```yaml
multimodal:
  enabled: true
```
Then, run the multimodal example script. This will use both the sample text document and the sample image.
```bash
python examples/multimodal_rag.py
```

## ⚙️ Advanced Usage

For production scenarios, re-indexing your documents on every query is inefficient. The library provides a two-step process for this:

1.  **Index Your Data**: Run a script to process and store your documents in a persistent vector store.
2.  **Query the Store**: Run your application to load the pre-indexed store and query it repeatedly.

<details>
<summary>Click to see an example of the advanced workflow</summary>

```python
# script_to_index.py
from easy_rag_pipeline import create_and_persist_vector_store, load_config

# Load config
config = load_config("examples/config.yaml")

# Create and save a FAISS vector store from a PDF
create_and_persist_vector_store(
    source_path="path/to/my_document.pdf",
    source_type="pdf",
    config=config,
    save_path="my_vector_store"
)

# --------------------------------------------------

# script_to_query.py
from easy_rag_pipeline import query_rag_pipeline, load_config
from langchain_community.vectorstores import FAISS
from easy_rag_pipeline.embed import get_embedding_function

# Load config and embedding function
config = load_config("examples/config.yaml")
embedding_function = get_embedding_function(config['embedding'])

# Load the persisted vector store
vector_store = FAISS.load_local("my_vector_store", embedding_function, allow_dangerous_deserialization=True)

# Query the pipeline
query = "What was the main finding of the document?"
answer = query_rag_pipeline(query, vector_store, config)
print(answer)
```

</details>

## 🔧 Extensibility

The library is designed to be easily extended.

-   **To add a new LLM**:
    1.  Install the required package (e.g., `pip install langchain-anthropic`).
    2.  Add an `elif provider == "anthropic":` block in `easy_rag_pipeline/generate.py`.
    3.  Update `config.py` to load the `ANTHROPIC_API_KEY`.
-   **To add a new Document Loader**:
    1.  Add the loader function (e.g., `load_docx`) in `easy_rag_pipeline/ingest.py`.
    2.  Update the `pipeline.py` functions to accept the new `source_type`.

## 🗺️ Roadmap

This project is under active development. Future enhancements include:

- [ ] Support for more vector databases (Chroma, Pinecone, Weaviate).
- [ ] Asynchronous pipeline for improved performance.
- [ ] Hybrid retrieval combining keyword search (BM25) and semantic search.
- [ ] Support for image and multi-modal RAG.
- [ ] Integration with RAG evaluation frameworks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss your ideas.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

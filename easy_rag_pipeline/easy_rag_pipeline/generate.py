from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_answer(query: str, docs: list, llm_config: dict):
    """
    Generates an answer using an LLM based on the query and retrieved documents.

    Args:
        query (str): The user's original query.
        docs (list): A list of retrieved documents to use as context.
        llm_config (dict): Configuration for the LLM.
            Example for OpenAI:
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "...",
                "temperature": 0.7
            }

    Returns:
        str: The generated answer.
    """
    provider = llm_config.get("provider", "openai").lower()

    if provider == "openai":
        llm = ChatOpenAI(
            model=llm_config.get("model", "gpt-4o-mini"),
            openai_api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature", 0.7)
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    context = "\n\n".join([d.page_content for d in docs])

    prompt_template = """
    Use the following context to answer the question at the end.
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Using LangChain Expression Language (LCEL) for the chain
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": query
    })

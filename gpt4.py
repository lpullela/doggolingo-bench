from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import numpy as np
import os

# load api key
with open("api_key.txt") as f:
    OPENAI_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY


def query_gpt4_with_retriever(
    model_type="regular",
    query="Explain quantum mechanics.",
    threshold=0.2,
    use_rag=True,
):
    """
    Query GPT-4 with a retriever, supporting both "mini" and "regular" model types.

    Parameters:
        model_type (str): Specify "mini" for GPT-4 Mini or "regular" for GPT-4.
        query (str): The query to process.

    Returns:
        str: The response from the model.
    """

    if model_type == "mini":
        model_name = "gpt-4o"
    else:
        model_name = "gpt-4"

    llm = ChatOpenAI(model=model_name, temperature=0)

    # reload faiss vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "retriever_store", embeddings, allow_dangerous_deserialization=True
    )

    docs_and_scores = vector_store.similarity_search_with_score(query, k=2)

    retrieved_docs = [doc for doc, score in docs_and_scores]
    similarity_scores = [score for doc, score in docs_and_scores]

    # filter... irrelevent documents have score <= 0.2
    filtered_docs = [
        doc
        for idx, doc in enumerate(retrieved_docs)
        if similarity_scores[idx] >= threshold
    ]

    # create an augmented? prompt
    if use_rag and filtered_docs:
        retrieved_text = "\n\n".join([doc.page_content for doc in filtered_docs])
        prompt = f"Using the following context:\n{retrieved_text}\n\nAnswer the question: {query}"
    else:
        prompt = (
            query  # use fall back if a) we dont want context or b) we dont find matches
        )

    response = llm.invoke(prompt)

    return response


if __name__ == "__main__":
    response = query_gpt4_with_retriever(
        model_type="mini", query="Interpret the word 'awoo'.", use_rag=False
    )
    print("Response:", response.content)

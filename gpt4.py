from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os

# Load the API key from a file
with open("api_key.txt") as f:
    OPENAI_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY


def query_gpt4_with_retriever(
    model_type="regular", query="Explain quantum mechanics.", threshold=0.7
):
    """
    Query GPT-4 with a retriever, supporting both "mini" and "regular" model types.

    Parameters:
        model_type (str): Specify "mini" for GPT-4 Mini or "regular" for GPT-4.
        query (str): The query to process.

    Returns:
        str: The response from the model.
    """

    # Choose the model based on model_type
    if model_type == "mini":
        model_name = "gpt-4o"  # Replace with actual mini model name
    else:
        model_name = "gpt-4"

    # Initialize the chosen model
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Reload the FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "retriever_store", embeddings, allow_dangerous_deserialization=True
    )

    # Query the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    retrieved_docs = retriever.invoke(query)

    # Filter documents by similarity score
    filtered_docs = [
        doc
        for doc in retrieved_docs
        if "similarity" in doc.metadata and doc.metadata["similarity"] >= threshold
    ]

    # Create the prompt based on retrieved context or fallback
    if filtered_docs:
        retrieved_text = "\n\n".join([doc.page_content for doc in filtered_docs])
        prompt = f"Using the following context:\n{retrieved_text}\n\nAnswer the question: {query}"
    else:
        prompt = query  # Use fallback prompt if no documents meet the threshold

    # Pass the prompt to the LLM
    response = llm.invoke(prompt)

    return response


# Example usage
if __name__ == "__main__":
    response = query_gpt4_with_retriever(
        model_type="mini", query="Explain the meaning of Doggolingo."
    )
    print("Response:", response)

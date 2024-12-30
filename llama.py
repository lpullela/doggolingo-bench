from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os


def query_llama_with_retriever(model_type="llama", query="Explain quantum mechanics."):
    """
    Query the Llama-3.3-70b-instruct model with a retriever.

    Parameters:
        model_type (str): Specify the model type (e.g., "llama").
        query (str): The query to process.

    Returns:
        str: The response from the model.
    """

    # Load the Llama model and tokenizer
    model_name = (
        "Llama-3.3-70b-instruct"  # Replace with your actual Llama model name or path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Create the Llama pipeline
    llama_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=0
    )

    # Reload the FAISS vector store
    embeddings = OpenAIEmbeddings()  # Adjust this if you're using custom embeddings
    vector_store = FAISS.load_local(
        "retriever_store", embeddings, allow_dangerous_deserialization=True
    )

    # Query the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    retrieved_docs = retriever.invoke(query)

    # Filter documents by similarity score
    threshold = 0.7
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

    # Generate a response using the Llama model
    response = llama_pipeline(prompt, max_length=512, num_return_sequences=1)
    return response[0]["generated_text"]


# Example usage
if __name__ == "__main__":
    response = query_llama_with_retriever(
        model_type="llama", query="Explain the meaning of Doggolingo."
    )
    print("Response:", response)

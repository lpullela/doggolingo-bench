from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import numpy as np
import os
import pandas as pd

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
        retrieved_text = filtered_docs[0].page_content
        prompt = (
            f"Using the true definition of the word:\n{retrieved_text}\n\nRate the proceeding content on the following scale: BAD, OKAY, GOOD, EXCELLENT. Content: "
            + query
        )
    else:
        prompt = (
            query  # use fall back if a) we dont want context or b) we dont find matches
        )

    response = llm.invoke(prompt)

    return response


if __name__ == "__main__":
    prompts = {
        "Interpret": "Interpret the word: ",
        "Create": "Create sentences containing the word: ",
        "Translate": "Translate the following sentences to formal English: ",
        "Generate": "Generate new words similar to the word: ",
    }

    df = pd.read_csv("data/doggolingo_dict.csv")
    results_df = pd.DataFrame()

    for word in df["word"]:
        responses = {}
        for i in prompts:
            q = prompts[i] + word
            if i == "Translate":
                q += responses["Create"]
            response = query_gpt4_with_retriever(
                model_type="mini", query=q, use_rag=False
            )
            responses[i] = response.content

        print("responses:", responses)

        entry = {"word": word}
        entry.update(responses)
        results_df = pd.concat([results_df, pd.DataFrame([entry])], ignore_index=True)

    # now we can test: using the augmented prompts, test whether the respnonses make sense
    for word in df["word"]:
        ratings = {}
        for p in prompts.keys():
            response_to_check = (
                prompts[p]
                + ":"
                + word
                + "\n"
                + results_df.loc[results_df["word"] == word, p].iloc[0]
            )
            query = f"{response_to_check}"

            response = query_gpt4_with_retriever(
                model_type="mini", query=query, use_rag=True
            )

            if "excellent" in response.content.lower():
                ratings[p] = "EXCELLENT"
            elif "good" in response.content.lower():
                ratings[p] = "GOOD"
            elif "okay" in response.content.lower():
                ratings[p] = "OKAY"
            elif "bad" in response.content.lower():
                ratings[p] = "BAD"
            else:
                ratings[p] = "unknown"

        for prompt_type, rating in ratings.items():
            column_name = f"{prompt_type}_rating"
            if column_name not in results_df.columns:
                results_df[column_name] = None
            results_df.loc[results_df["word"] == word, column_name] = rating

    results_df.to_csv("responses_output.csv", index=False)

    print("Responses saved to responses_output.csv")

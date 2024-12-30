from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import os

# loads the retriever and prompts using gpt 4.0

with open("api_key.txt") as f:
    OPENAI_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# using gpt 4
llm = ChatOpenAI(model="gpt-4", temperature=0)

# reload the faiss vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(
    "retriever_store", embeddings, allow_dangerous_deserialization=True
)

# query the retriever

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retrieved_docs = retriever.invoke("Explain quantum mechanics.")

# filter documents by similarity score (assuming similarity score is stored in the metadata)
threshold = 0.7
filtered_docs = [
    doc
    for doc in retrieved_docs
    if "similarity" in doc.metadata and doc.metadata["similarity"] >= threshold
]

# combine retrieved text into the LLM prompt
if filtered_docs:
    retrieved_text = "\n\n".join([doc.page_content for doc in filtered_docs])
    prompt = f"Using the following context:\n{retrieved_text}\n\nAnswer the question: What is quantum mechanics?"  # should be explain the meaning of {doggolingo}
else:
    prompt = "Explain quantum mechanics."  # use the original prompt

# Pass the prompt to the LLM
response = llm.invoke(prompt)

# Print the results
print("Prompt:", prompt)
print("Response:", response)

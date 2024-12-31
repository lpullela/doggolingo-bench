from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

# this file gets the dictionary of doggolingo terms into a retriever
# only run this once to generate the retriever store file
# before running this first we will need to compile a dict of DL terms and their 'definition'

with open("api_key.txt") as f:
    OPENAI_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
embeddings = OpenAIEmbeddings()


class LineTextSplitter:
    def create_documents(self, file_path):
        """
        Parses the given text file line by line and creates a list of documents.
        Each line in the file will be treated as a separate document.
        """
        documents = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    documents.append(Document(page_content=line))
        return documents


# split dictionary document into individual words
text_file_path = "data/doggolingo_dict.csv"
line_text_splitter = LineTextSplitter()
documents = line_text_splitter.create_documents(text_file_path)

# embed/store
doc_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])
vector_store = FAISS.from_documents(documents, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# # llm with retrieval
# qa_chain = RetrievalQA.from_chain_type(
#     llm=OpenAI(model="gpt-4"), retriever=retriever, return_source_documents=True
# )

# example usage

# # augment the prompt
# query = "What is the summary of the document?"
# response = qa_chain.run(query)

# # Response with source documents
# print(response["answer"])
# print(response["source_documents"])

# this is what we will be doing:

# # augment llm prompt example usage
# retrieved_docs = retriever.get_relevant_documents("Explain quantum mechanics.")

# # Combine retrieved text into the LLM prompt
# retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
# prompt = f"Using the following context:\n{retrieved_text}\n\nAnswer the question: What is quantum mechanics?"

# # Pass the prompt to the LLM
# response = llm(prompt)
# print(response)

vector_store.save_local("retriever_store")

# loading the retriever
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings

# # Reload the FAISS vector store
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# vector_store = FAISS.load_local("retriever_store", embeddings)

# # Create the retriever
# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

print("done")

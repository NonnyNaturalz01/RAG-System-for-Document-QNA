from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

# Load embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Retriever with k=5
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

# Query
query = "Who is the author of Dracula?"
docs = retriever.invoke(query)

print(f"User Query: {query}")
print("content")
for i, doc in enumerate(docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")




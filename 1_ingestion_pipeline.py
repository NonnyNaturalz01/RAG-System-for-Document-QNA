import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_files(file_path="docs"):
    #Loading all text files from the docs directory
    print(f"Loading documents from {file_path}")

    #check if docs directory exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} directory not found. Please create the directory and add text files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=file_path, 
        glob="*.txt", 
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
        
    )
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No text files found in {file_path} directory. Please add text files to the directory.")
    
    for i, doc in enumerate(documents[:2]):#show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.page_content[:500]}")  # first 500 characters
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")  # first 100 characters
        print(f"Metadata: {doc.metadata}")

        return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    #Splitting documents into chunks
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    #Creating a ChromeDB vector store from the chunks
    print(f"Creating ChromeDBvector store in {persist_directory}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
)


    # Create Chroma vector store
    print("creating vector store ")
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore

def main():
    print("Main Function")

    #step 1: load the files 
    documents = load_files(file_path="docs")

    #step 2: chunk the files
    chunks=split_documents(documents)

    # step 3: embedding and storing in vector database
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()



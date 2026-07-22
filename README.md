# 📚 Retrieval-Augmented Generation (RAG) System for Document Q&A
<br>
<br>
This project implements a <b>RAG pipeline</b> for answering questions from documents using modern NLP techniques.  
It combines <b>HuggingFace embeddings</b>, <b>ChromaDB vector storage</b>, and a <b>CharacterTextSplitter chunking strategy</b> to deliver accurate, context-aware responses.
<br>
---
<br>
<br>
 ✅ Key Components

- **[HuggingFace Embeddings](ca://s?q=HuggingFace_embeddings_for_RAG)**  
  Converts text chunks into dense vector representations for semantic similarity search.

- **[ChromaDB](ca://s?q=ChromaDB_vector_database_for_RAG)**  
  Stores embeddings and enables efficient retrieval of relevant document chunks.

- **[CharacterTextSplitter](ca://s?q=CharacterTextSplitter_chunking_strategy)**  
  Splits documents into manageable chunks, ensuring context is preserved during retrieval.

- **[Ingestion Pipeline](ca://s?q=RAG_ingestion_pipeline)**  
  Processes raw documents, applies chunking, generates embeddings, and stores them in ChromaDB.

- **[Retrieval Pipeline](ca://s?q=RAG_retrieval_pipeline)**  
  Retrieves the most relevant chunks based on user queries and feeds them into the language model for answer generation.
<br>
---
<br>
<br>
## 📂 Repository Structure

- `ingestion.py` → Handles document preprocessing, chunking, and embedding storage.  
- `retrieval.py` → Manages query embedding, similarity search, and response generation.  
- `requirements.txt` → Dependencies for HuggingFace, ChromaDB, and supporting libraries.  
- `README.md` → Project documentation.

---


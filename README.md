# HackRx 6.0 - Intelligent Document Q&A System

This project is a production-ready, LLM-powered system built for the HackRx 6.0 hackathon. It processes large, unstructured documents (like insurance policies) and answers natural language questions based on their content.

## Problem Statement

The challenge was to build a system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails. The system must be accurate, explainable, and efficient.

## Our Solution & Architecture

We have implemented a robust **Retrieval-Augmented Generation (RAG)** pipeline to meet the challenge. The system operates in a multi-step process to ensure accuracy and traceability:

1.  **Document Ingestion:** The system accepts a URL to a PDF or DOCX file, loads the document, and extracts the raw text.
2.  **Text Chunking & Embedding:** The extracted text is split into smaller, semantically meaningful chunks. Each chunk is then converted into a numerical vector (embedding) using the `all-MiniLM-L6-v2` model.
3.  **Vector Indexing:** These embeddings are stored and indexed in a FAISS vector database for efficient semantic search.
4.  **Multi-Query Processing:** The system receives an array of questions. For each question, it performs a semantic search against the indexed document chunks to find the most relevant context.
5.  **LLM-Powered Answer Generation:** The retrieved context and the original question are passed to Google's **Gemini 1.5 Flash** model. A carefully engineered prompt instructs the model to act as an expert Q&A assistant and formulate a direct, concise answer based *only* on the provided text.
6.  **API Delivery:** The entire system is wrapped in a **FastAPI** backend, providing a secure, high-performance API endpoint (`/hackrx/run`) as required by the hackathon specifications.

## Tech Stack

- **Backend:** FastAPI
- **LLM:** Google Gemini 1.5 Flash
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS (Facebook AI Similarity Search)
- **Document Parsing:** `pypdf`, `python-docx`
- **Text Splitting:** `langchain`

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set your API Key:**
    Open `main.py` and replace `"YOUR_GEMINI_API_KEY"` with your actual API key from Google AI Studio.
4.  **Run the server:**
    ```bash
    uvicorn main:app --reload
    ```
5.  **Access the interactive documentation** at `http://127.0.0.1:8000/docs` to test the API.
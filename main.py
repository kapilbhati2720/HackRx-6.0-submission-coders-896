import requests
import io
import json
import os
from typing import List

# --- Core Libraries ---
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import faiss
import numpy as np

# --- Document Processing ---
from pypdf import PdfReader
from docx import Document

# --- AI & Embeddings ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- Configuration ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual key
os.environ['GOOGLE_API_KEY'] = "AIzaSyARPqQzg79gAVbtmvgAFyblhPO1055nUuk"
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# --- Component 1: Document Ingestion & Chunking (Unchanged) ---

def get_document_text(url: str) -> str:
    """Fetches and extracts text from a PDF or DOCX file at a given URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type')
        if 'pdf' in content_type:
            with io.BytesIO(response.content) as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages)
        elif 'openxmlformats-officedocument.wordprocessingml.document' in content_type:
            with io.BytesIO(response.content) as f:
                doc = Document(f)
                text = "\n".join(para.text for para in doc.paragraphs)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}.")
        return text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch document from URL: {e}")

def get_text_chunks(text: str) -> list[str]:
    """Splits text into smaller, semantically manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# --- Component 2: Vector Store (Unchanged) ---

class VectorStore:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        self.model_name = model_name
        self.model = None  # Initialize model as None
        self.index = None
        self.chunks = []

    def _load_model(self):
        """Loads the model if it hasn't been loaded yet."""
        if self.model is None:
            print("Loading embedding model for the first time...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully.")

    def build_index(self, chunks: list[str]):
        self._load_model()
        """Creates a FAISS index from text chunks."""
        self.chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_tensor=False)
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.array(embeddings))
        print(f"FAISS index built successfully with {len(chunks)} chunks.")

    def search(self, query: str, k: int = 5) -> list[str]:
        self._load_model()
        """Performs a semantic search."""
        if self.index is None: return []
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k)
        return [self.chunks[i] for i in indices[0]]

# --- Component 3: LLM Q&A Function (New) ---

def get_answer_with_llm(question: str, context: list[str]) -> str:
    """Generates a direct answer to a question based on context."""
    context_str = "\n\n---\n\n".join(context)
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    You are an expert Q&A assistant for policy documents.
    Your task is to answer the user's question based *only* on the provided text excerpts.
    Be concise and directly answer the question. If the information is not in the excerpts,
    state that the answer is not found in the provided text.

    **CONTEXT EXCERPTS:**
    ---
    {context_str}
    ---

    **QUESTION:**
    "{question}"

    **ANSWER:**
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error during LLM call for question '{question}': {e}")
        return "Error generating answer from LLM."

# --- Component 4: FastAPI Application (Corrected for Auth Button) ---

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security

app = FastAPI(
    title="HackRx 6.0 Q&A System",
    description="An LLM-powered system to answer questions on documents."
)

# NEW: Pydantic models matching the submission guide
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

vector_store = VectorStore()

# NEW: Explicitly define the security scheme
auth_scheme = HTTPBearer()

# REFACTORED: The main endpoint logic with corrected dependency
@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(request: QARequest, token: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """
    The main Q&A endpoint that receives a document URL and a list of questions,
    and returns a list of answers. Requires Bearer token authentication.
    """
    # The 'token' variable now holds the credential, but we don't need to use it for local testing.
    # The platform will use it for real evaluation.
    
    try:
        # Step 1: Document Ingestion and Indexing
        print(f"Fetching document from: {request.documents}")
        document_text = get_document_text(request.documents)
        chunks = get_text_chunks(document_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from document.")
        vector_store.build_index(chunks)
        
        # Step 2: Loop through questions and generate answers
        all_answers = []
        for question in request.questions:
            print(f"Processing question: '{question}'")
            # Step 2a: Retrieve context for the current question
            retrieved_chunks = vector_store.search(question, k=5)
            if not retrieved_chunks:
                all_answers.append("No relevant information found in the document to answer this question.")
                continue
            
            # Step 2b: Generate answer with LLM
            answer = get_answer_with_llm(question, retrieved_chunks)
            all_answers.append(answer)
            print(f"Generated answer: '{answer}'")
        
        return QAResponse(answers=all_answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

        raise HTTPException(status_code=500, detail=str(e))


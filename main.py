import os
import io
import json
from typing import List

# --- Core Libraries ---
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests
import numpy as np

# --- Document Processing ---
from pypdf import PdfReader
from docx import Document

# --- AI & Embeddings ---
import google.generativeai as genai

# --- Pinecone Client ---
from pinecone import Pinecone

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hackrx-retrieval"

# --- LAZY INITIALIZATION ---
pc = None
pinecone_index = None
genai_configured = False

def initialize_dependencies():
    """Initializes clients on the first request."""
    global pc, pinecone_index, genai_configured
    # This function is now called only by the main endpoint, not the health check
    if not genai_configured:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            genai_configured = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to configure Google AI: {e}")
    
    if pc is None:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize Pinecone: {e}")

# --- All other functions (embed_text, get_document_text, etc.) remain the same ---
def embed_text_with_gemini(text: List[str], task_type: str) -> List[List[float]]:
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        return [[] for _ in text]

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_document_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        content_type = response.headers.get('content-type')
        if 'pdf' in content_type:
            with io.BytesIO(response.content) as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        else:
            with io.BytesIO(response.content) as f:
                doc = Document(f)
                text = "\n".join(para.text for para in doc.paragraphs)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse document: {e}")

def get_batch_answers_with_llm(questions: List[str], context: str) -> List[str]:
    model = genai.GenerativeModel('gemini-1.5-flash')
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""
    Based ONLY on the provided "CONTEXT", answer each of the "QUESTIONS".
    Your response must be a single JSON object with a key "answers", which is a JSON array of strings.
    The array must contain exactly {len(questions)} answers in the same order as the questions.
    If the answer to a question is not found, respond with "Answer not found in the provided document."

    CONTEXT:
    ---
    {context}
    ---
    QUESTIONS:
    {formatted_questions}
    JSON RESPONSE:
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(json_str)
        return result.get("answers", [])
    except Exception as e:
        return [f"Error processing batch: {e}" for _ in questions]

# --- FastAPI Application ---
app = FastAPI(title="HackRx 6.0 Q&A System")

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

auth_scheme = HTTPBearer()

# --- NEW HEALTH CHECK ENDPOINT ---
@app.get("/healthz")
def health_check():
    """A simple endpoint that Render can ping to confirm the service is alive."""
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(request: QARequest, token: HTTPAuthorizationCredentials = Security(auth_scheme)):
    # Initialize dependencies on the first request to the main endpoint
    initialize_dependencies()

    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone index could not be initialized.")
    try:
        document_text = get_document_text(request.documents)
        chunks = split_text_into_chunks(document_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        chunk_embeddings = embed_text_with_gemini(text=chunks, task_type="RETRIEVAL_DOCUMENT")

        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            if embedding:
                vectors_to_upsert.append({
                    "id": f"chunk_{i}",
                    "values": embedding,
                    "metadata": {"text": chunk}
                })
        if vectors_to_upsert:
             pinecone_index.upsert(vectors=vectors_to_upsert, namespace=request.documents)

        comprehensive_context = set()
        question_embeddings = embed_text_with_gemini(text=request.questions, task_type="RETRIEVAL_QUERY")
        
        for q_embedding in question_embeddings:
            if not q_embedding: continue
            query_results = pinecone_index.query(
                namespace=request.documents,
                vector=q_embedding,
                top_k=3,
                include_metadata=True
            )
            for match in query_results['matches']:
                comprehensive_context.add(match['metadata']['text'])
        
        context_str = "\n".join(list(comprehensive_context))

        all_answers = get_batch_answers_with_llm(request.questions, context_str)
        
        return QAResponse(answers=all_answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

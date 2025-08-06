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
# These keys MUST be set in Render's Environment Variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hackrx-retrieval"

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print("Dependencies initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")
    pinecone_index = None

# --- Gemini Embedding Function ---
def embed_text_with_gemini(text: List[str], task_type: str) -> List[List[float]]:
    try:
        # Using Google's text-embedding-004 model which has a dimension of 768
        result = genai.embed_content(model="models/text-embedding-004",
                                     content=text,
                                     task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"Error creating embeddings with Gemini: {e}")
        return [[] for _ in text]

# --- Text & Document Processing ---
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
        response = requests.get(url, timeout=30)
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

# --- Q&A Function ---
def get_batch_answers_with_llm(questions: List[str], context: str) -> List[str]:
    model = genai.GenerativeModel('gemini-1.5-flash')
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""
    You are an expert Q&A assistant for policy documents.
    Based *only* on the provided "CONTEXT", answer each of the "QUESTIONS" listed below.
    Your response must be a single JSON object with a key "answers", which is a JSON array of strings.
    The array must contain exactly {len(questions)} string answers, in the same order as the questions.
    If the answer to a specific question is not found in the context, the string for that answer should be "Answer not found in the provided document."

    CONTEXT:
    ---
    {context}
    ---

    QUESTIONS:
    {formatted_questions}

    JSON RESPONSE FORMAT:
    {{
      "answers": [
        "Answer to question 1",
        "Answer to question 2",
        ...
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(json_str)
        return result.get("answers", [])
    except Exception as e:
        print(f"Error during batch LLM call: {e}")
        return [f"Error processing batch: {e}" for _ in questions]

# --- FastAPI Application ---
app = FastAPI(title="HackRx 6.0 Q&A System")

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

auth_scheme = HTTPBearer()

@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(request: QARequest, token: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if not pinecone_index or not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Server is not configured correctly.")
    try:
        # Step 1: Ingest and chunk document
        document_text = get_document_text(request.documents)
        chunks = split_text_into_chunks(document_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        # Step 2: Create embeddings for chunks using Gemini
        print(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings = embed_text_with_gemini(text=chunks, task_type="RETRIEVAL_DOCUMENT")

        # Step 3: Upsert embeddings AND metadata to Pinecone
        print("Upserting to Pinecone...")
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            if embedding:
                vectors_to_upsert.append({
                    "id": f"chunk_{i}",
                    "values": embedding, # This is the required field
                    "metadata": {"text": chunk}
                })
        if vectors_to_upsert:
             pinecone_index.upsert(vectors=vectors_to_upsert, namespace=request.documents)

        # Step 4: For each question, embed it and query Pinecone
        comprehensive_context = set()
        print(f"Generating embeddings for {len(request.questions)} questions...")
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
        
        context_str = "\n\n---\n\n".join(list(comprehensive_context))

        # Step 5: Get answers from LLM
        all_answers = get_batch_answers_with_llm(request.questions, context_str)
        
        return QAResponse(answers=all_answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


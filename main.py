import requests
import io
import os
import json
from typing import List

# --- Core Libraries ---
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import numpy as np

# --- Document Processing ---
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- AI & Embeddings ---
import google.generativeai as genai

# --- Configuration ---
os.environ['GOOGLE_API_KEY'] = "AIzaSyARPqQzg79gAVbtmvgAFyblhPO1055nUuk"
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# --- NEW: Simplified Logic using Gemini Embeddings ---

def embed_text_with_gemini(text: List[str]) -> List[List[float]]:
    """Uses the Gemini API to create embeddings for a list of text chunks."""
    try:
        # Using the new text-embedding-004 model from Google
        result = genai.embed_content(model="models/text-embedding-004",
                                     content=text,
                                     task_type="RETRIEVAL_DOCUMENT")
        return result['embedding']
    except Exception as e:
        print(f"Error creating embeddings with Gemini: {e}")
        return [[] for _ in text]

def get_document_text(url: str) -> str:
    # This function remains the same
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type')
        if 'pdf' in content_type:
            with io.BytesIO(response.content) as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages)
        else: # Assuming DOCX if not PDF
            with io.BytesIO(response.content) as f:
                doc = Document(f)
                text = "\n".join(para.text for para in doc.paragraphs)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse document: {e}")

def get_text_chunks(text: str) -> list[str]:
    # This function remains the same
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_batch_answers_with_llm(questions: List[str], context: str) -> List[str]:
    # This function and its prompt remain the same
    model = genai.GenerativeModel('gemini-1.5-flash')
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""
    You are an expert Q&A assistant for policy documents.
    Based *only* on the provided "CONTEXT", answer each of the "QUESTIONS" listed below.
    Your response must be a single JSON object with a key "answers", which is a JSON array of strings.
    The array must contain exactly {len(questions)} string answers, in the same order as the questions.
    If the answer to a specific question is not found in the context, the string for that answer should be "Answer not found in the provided document."

    **CONTEXT:**
    ---
    {context}
    ---

    **QUESTIONS:**
    {formatted_questions}

    **JSON RESPONSE FORMAT:**
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
    try:
        # Step 1: Ingest and chunk document
        document_text = get_document_text(request.documents)
        chunks = get_text_chunks(document_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from document.")
        
        # Step 2: Get embeddings for all chunks from Gemini API
        print(f"Generating embeddings for {len(chunks)} document chunks...")
        chunk_embeddings = np.array(embed_text_with_gemini(chunks))
        
        # Step 3: Get embeddings for all questions
        print(f"Generating embeddings for {len(request.questions)} questions...")
        question_embeddings = np.array(embed_text_with_gemini(request.questions))
        
        # Step 4: For each question, find the most relevant chunks (manual similarity search)
        comprehensive_context = set()
        for q_embedding in question_embeddings:
            # Calculate cosine similarity
            similarities = np.dot(chunk_embeddings, q_embedding) / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding))
            # Get top 3 chunks
            top_indices = np.argsort(similarities)[-3:][::-1]
            for index in top_indices:
                comprehensive_context.add(chunks[index])
        
        context_str = "\n\n---\n\n".join(list(comprehensive_context))
        
        # Step 5: Make a single batch call to the LLM
        all_answers = get_batch_answers_with_llm(request.questions, context_str)
        
        return QAResponse(answers=all_answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


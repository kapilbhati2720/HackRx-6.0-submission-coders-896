import os
import io
import json
from typing import List
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests
import numpy as np
from pypdf import PdfReader
from docx import Document
import google.generativeai as genai
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hackrx-retrieval"

# --- LAZY INITIALIZATION ---
pinecone_index = None
genai_configured = False

def initialize_dependencies():
    global pinecone_index, genai_configured
    if not genai_configured:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            genai_configured = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to configure Google AI: {e}")
    if pinecone_index is None:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize Pinecone: {e}")

# --- HELPER FUNCTIONS ---
def embed_questions_with_gemini(text: List[str]) -> List[List[float]]:
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="RETRIEVAL_QUERY")
        return result['embedding']
    except Exception as e:
        print(f"Error creating question embeddings: {e}")
        return [[] for _ in text]

def get_batch_answers_with_llm(questions: List[str], context: str) -> List[str]:
    model = genai.GenerativeModel('gemini-1.5-flash')
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""
    Based ONLY on the provided "CONTEXT", intelligently answer each of the "QUESTIONS".
    Your response must be a single JSON object with a key "answers", which is a JSON array of strings.
    The array must contain exactly {len(questions)} answers in the same order as the questions.
    Synthesize the information from the context to form a concise, direct answer.
    If the answer cannot be determined from the context, respond with "Answer not found in the provided document."

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
        answers = result.get("answers", [])
        # Ensure the list has the correct number of answers
        if len(answers) != len(questions):
            return ["LLM did not return the correct number of answers." for _ in questions]
        return answers
    except Exception as e:
        print(f"Error during batch LLM call: {e}")
        return [f"Error processing in LLM: {e}" for _ in questions]

# --- FASTAPI APP ---
app = FastAPI(title="HackRx 6.0 Q&A System")

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

auth_scheme = HTTPBearer()

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(request: QARequest, token: HTTPAuthorizationCredentials = Security(auth_scheme)):
    initialize_dependencies()

    try:
        # Step 1: Embed the incoming questions
        question_embeddings = embed_questions_with_gemini(request.questions)
        
        # Step 2: Query Pinecone for context for all questions
        comprehensive_context = set()
        for q_embedding in question_embeddings:
            if not q_embedding: continue
            query_results = pinecone_index.query(
                namespace=request.documents,
                vector=q_embedding,
                top_k=3, # Retrieve top 3 chunks per question
                include_metadata=True
            )
            for match in query_results['matches']:
                comprehensive_context.add(match['metadata']['text'])
        
        context_str = "\n\n---\n\n".join(list(comprehensive_context))
        
        if not comprehensive_context:
            return QAResponse(answers=["No relevant context found in the indexed documents." for _ in request.questions])
            
        # Step 3: Use the retrieved context to GENERATE intelligent answers with the LLM
        all_answers = get_batch_answers_with_llm(request.questions, context_str)
        
        return QAResponse(answers=all_answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

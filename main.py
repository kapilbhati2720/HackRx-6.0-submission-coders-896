import os
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
from sentence_transformers.cross_encoder import CrossEncoder

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hackrx-retrieval"

# --- LAZY INITIALIZATION ---
pinecone_index = None
genai_configured = False
cross_encoder = None

def initialize_dependencies():
    global pinecone_index, genai_configured, cross_encoder
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
    if cross_encoder is None:
        try:
            # Using a very lightweight CrossEncoder model suitable for CPU
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load CrossEncoder model: {e}")

# --- HELPER FUNCTIONS ---
def embed_questions_with_gemini(text: List[str]) -> List[List[float]]:
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="RETRIEVAL_QUERY")
        return result['embedding']
    except Exception as e:
        return [[] for _ in text]

def get_batch_answers_with_llm(questions: List[str], context: str) -> List[str]:
    model = genai.GenerativeModel('gemini-1.5-flash')
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""
    Based ONLY on the provided "CONTEXT", intelligently answer each of the "QUESTIONS".
    Synthesize the information to form a concise, direct answer.
    If the answer cannot be determined from the context, respond with "Answer not found in the provided document."

    CONTEXT:
    ---
    {context}
    ---
    QUESTIONS:
    {formatted_questions}
    JSON RESPONSE (must be a JSON object with a single key "answers" which is an array of strings):
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(json_str)
        return result.get("answers", [])
    except Exception as e:
        return [f"Error in LLM: {e}" for _ in questions]

# --- FASTAPI APP ---
app = FastAPI(title="HackRx 6.0 Q&A System")
# ... (QARequest, QAResponse, auth_scheme, healthz models are the same) ...

@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(request: QARequest, token: HTTPAuthorizationCredentials = Security(auth_scheme)):
    initialize_dependencies()

    try:
        question_embeddings = embed_questions_with_gemini(request.questions)
        
        all_answers = []
        for i, q_embedding in enumerate(question_embeddings):
            question = request.questions[i]
            if not q_embedding:
                all_answers.append("Failed to generate embedding for this question.")
                continue

            # 1. RETRIEVE a broad set of documents
            retrieved_docs = pinecone_index.query(
                namespace=request.documents,
                vector=q_embedding,
                top_k=10, # Retrieve more documents for re-ranking
                include_metadata=True
            )
            
            if not retrieved_docs['matches']:
                all_answers.append("No relevant context found in initial retrieval.")
                continue

            # 2. RE-RANK the results for higher accuracy
            cross_input = [[question, match['metadata']['text']] for match in retrieved_docs['matches']]
            cross_scores = cross_encoder.predict(cross_input)
            
            # Combine scores with documents and sort
            doc_scores = list(zip(retrieved_docs['matches'], cross_scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 3. GENERATE the answer using the best re-ranked context
            final_context = "\n\n---\n\n".join([doc['metadata']['text'] for doc, score in doc_scores[:3]]) # Use top 3 re-ranked docs
            
            answer = get_batch_answers_with_llm([question], final_context)
            all_answers.append(answer[0] if answer else "Failed to generate answer.")

        return QAResponse(answers=all_answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

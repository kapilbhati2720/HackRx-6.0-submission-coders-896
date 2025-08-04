import os
import io
from typing import List

# --- Core Libraries ---
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests

# --- Document Processing ---
from pypdf import PdfReader
from docx import Document

# --- Pinecone Client ---
from pinecone import Pinecone

# --- Configuration ---
# These will be set in Render's Environment Variables
os.environ['GOOGLE_API_KEY'] = "AIzaSyARPqQzg79gAVbtmvgAFyblhPO1055nUuk" # For the Q&A part
PINECONE_API_KEY = os.environ.get("pcsk_4EqDAc_TNFrAcV3hQHfiB3rB79JGHcJ39QzkM8eujmRbAeUFPiNifqsBS7BQ3KHPShUnqY")

# --- Initialize Pinecone ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("hackrx-retrieval") # The name of the index you just created
    print("Pinecone initialized successfully.")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    pinecone_index = None

# --- Text & Document Processing ---
def split_text_into_chunks(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
    # Simple text splitter
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_document_text(url: str) -> str:
    # Unchanged
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type')
        if 'pdf' in content_type:
            with io.BytesIO(response.content) as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        else: # Assuming DOCX
            with io.BytesIO(response.content) as f:
                doc = Document(f)
                text = "\n".join(para.text for para in doc.paragraphs)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse document: {e}")

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
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone index is not available.")

    try:
        # Step 1: Ingest and chunk document
        print(f"Processing document: {request.documents}")
        document_text = get_document_text(request.documents)
        chunks = split_text_into_chunks(document_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        # Step 2: Upsert document chunks to Pinecone. Pinecone will create the embeddings via llama-text-embed-v2
        print(f"Upserting {len(chunks)} chunks to Pinecone...")
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
             vectors_to_upsert.append({
                 "id": f"chunk_{i}",
                 "metadata": {
                     # Pinecone's model will use this 'text' field to create the embedding
                     "text": chunk
                 }
             })
        pinecone_index.upsert(vectors=vectors_to_upsert, namespace=request.documents)

        # Step 3: For each question, query Pinecone to get relevant context
        # In this modern Pinecone setup, we don't need a separate Q&A LLM call.
        # We can get answers directly from the retrieved context.
        all_answers = []
        for question in request.questions:
            print(f"Querying for question: {question}")
            # Pinecone creates the embedding for our query text automatically
            # We pass the question directly to the 'text' field for embedding
            query_results = pinecone_index.query(
                namespace=request.documents,
                top_k=1, # Get the single best matching chunk
                include_metadata=True,
                # The text to be embedded and searched for is passed here
                text=question
            )
            
            if query_results['matches']:
                # The most relevant text chunk is our answer
                answer = query_results['matches'][0]['metadata']['text']
                all_answers.append(answer)
            else:
                all_answers.append("No relevant information found in the document.")

        return QAResponse(answers=all_answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


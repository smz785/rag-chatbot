from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.pipeline import RAGPipeline

app = FastAPI(title="Rag Chatbot (Ollama + FastAPI)")
rag = RAGPipeline()

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"health": "ok"}

@app.post("/ask")
def ask(request: AskRequest):
    return rag.ask(request.question)
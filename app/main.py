from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.pipeline import RAGPipeline
from contextlib import asynccontextmanager
import uvicorn

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.rag = RAGPipeline()
    yield

app = FastAPI(title="Rag Chatbot (Ollama + FastAPI)", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {'message': 'Hello, go to /docs'}

@app.get("/health")
def health():
    return {"status": "ok", "rag loaded": hasattr(app.state,'rag')}

@app.post("/ask")
def ask(request: AskRequest):
    rag: RAGPipeline = app.state.rag
    return rag.ask(request.question)

if __name__ == '__main__':
    uvicorn.run('main:app',host='127.0.0.1', port=8000, reload=True)
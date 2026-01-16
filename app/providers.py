from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.config import Settings

def get_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )

def get_chatbot_llm(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


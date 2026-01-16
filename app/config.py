from dataclasses import dataclass
import os

def _env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val == '' or val is None:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val

def _env_int(name:str, default: int) -> int:
    raw = os.getenv(name, default)
    if raw is None or raw == "":
        return default

    try:
        return int(raw)
    except Exception as e:
        raise ValueError(f"Invalid int data type for environment variable: {name}: {e}")

@dataclass(frozen=True)
class Settings:
   llm_provider: str
   ollama_base_url: str
   ollama_chat_model: str
   ollama_embed_model: str

   rag_top_k: int
   chunk_size: int
   chunk_overlap: int
   index_dir: str
   pdf_dir: str

def load_settings() -> Settings:
    llm_provider = _env("LLM_PROVIDER", 'ollama').lower()
    if llm_provider != 'ollama':
        raise RuntimeError(f"This code version only supports {llm_provider}")
    return Settings(
        llm_provider = llm_provider,
        ollama_base_url = _env("OLLAMA_BASE_URL", 'OLLAMA_BASE'),
        ollama_chat_model = _env("OLLAMA_CHAT_MODEL", 'llama3.1'),
        ollama_embed_model = _env("OLLAMA_EMBED_MODEL", 'nomic-embed-text'),
        rag_top_k = _env_int("RAG_TOP_K", 8),
        chunk_size = _env_int("CHUNK_SIZE", 800),
        chunk_overlap = _env_int("CHUNK_OVERLAP", 120),
        index_dir = _env("INDEX_DIR", 'app/indexes/faiss'),
        pdf_dir = _env("PDF_DIR", 'app/data/pdfs')
    )
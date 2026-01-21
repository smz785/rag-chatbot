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
   # routing
   doc_index_dir: str
   doc_route_top_n: int # how many docs to route to
   chunk_fetch_k: int # retrieve this many chunks before filtering
   doc_text_max_chars: int # how much text to use to represent each doc


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
        index_dir = _env("INDEX_DIR","indexes/chunk_idx"),
        pdf_dir = _env("PDF_DIR", "data/pdfs"),
        # routing
        doc_index_dir = _env("DOC_INDEX_DIR", "indexes/doc_idx"),
        doc_route_top_n = _env_int("DOC_ROUTE_TOP_N", 3), # how many docs to route to
        chunk_fetch_k = _env_int("CHUNK_FETCH_K", 40),  # retrieve this many chunks before filtering
        doc_text_max_chars = _env_int("DOC_TEXT_MAX_CHARS",12000),  # how much text to use to represent each doc

    )
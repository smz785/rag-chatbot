# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, LangChain, and Ollama. This chatbot uses FAISS vector stores to search through PDF documents and provides contextually relevant answers with source citations.

## Features

-  **LLM-powered Q&A** using Ollama (llama3.2, phi3, etc.)
-  **PDF document ingestion** with intelligent chunking
-  **Two-stage retrieval**: Document routing + chunk-level retrieval
-  **Source attribution** with page numbers and snippets
-  **FastAPI REST API** with async support
-  **Evaluation tools** for testing retrieval and answer quality

## Tech Stack

- **Backend Framework**: FastAPI
- **LLM Provider**: Ollama (local LLM inference)
- **Embeddings**: nomic-embed-text
- **Vector Store**: FAISS
- **Document Processing**: LangChain, PyPDF
- **Language**: Python 3.12+

## Project Structure

```
rag-chatbot/
├── app/                          # Main application code
│   ├── config.py                 # Configuration and environment variables
│   ├── main.py                   # FastAPI application entry point
│   ├── providers.py              # LLM and embedding provider setup
│   └── rag/                      # RAG pipeline implementation
│       ├── pipeline.py           # Core RAG logic with two-stage retrieval
│       └── prompts.py            # System prompts for LLM
│
├── ingests/                      # Document ingestion and indexing
│   ├── ingest.py                 # Main ingestion script
│   ├── loaders.py                # PDF loading utilities
│   └── chunking.py               # Document chunking strategies
│
├── data/                         # Data directory
│   └── pdfs/                     # Place your PDF files here
│
├── indexes/                      # FAISS vector indexes
│   ├── chunk_idx/                # Chunk-level retrieval index
│   └── doc_idx/                  # Document-level routing index
│
├── eval/                         # Evaluation scripts
│   ├── eval_answers.py           # Answer quality evaluation
│   ├── eval_retrieval.py         # Retrieval performance evaluation
│   └── questions.jsonl           # Test questions
│
├── tests/                        # Test files
├── requirements.txt              # Python dependencies
├── .env.example                  # Example environment configuration
└── README.md                     # This file
```

## Prerequisites

1. **Python 3.12+** installed
2. **Ollama** installed and running locally
   - Download from: https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull llama3.2:3b
     ollama pull nomic-embed-text
     ```

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```cmd
   cd C:\Users\SyedZain\PycharmProjects\rag-chatbot
   ```

2. **Create a virtual environment** (recommended):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
     ```cmd
     copy .env.example .env
     ```
   - Edit `.env` and update the following:
     ```env
     LLM_PROVIDER=ollama
     OLLAMA_BASE_URL=http://localhost:11434
     OLLAMA_CHAT_MODEL=llama3.2:3b
     OLLAMA_EMBED_MODEL=nomic-embed-text
     
     # RAG Configuration (optional - defaults shown)
     RAG_TOP_K=4
     CHUNK_SIZE=800
     CHUNK_OVERLAP=120
     INDEX_DIR=indexes/chunk_idx
     PDF_DIR=data/pdfs
     DOC_INDEX_DIR=indexes/doc_idx
     DOC_ROUTE_TOP_N=3
     CHUNK_FETCH_K=40
     DOC_TEXT_MAX_CHARS=12000
     ```

## Usage

### Step 1: Ingest Documents

Place your PDF files in the `data/pdfs/` directory, then run the ingestion script:

```cmd
python -m ingests.ingest
```

This will:
- Load all PDFs from `data/pdfs/`
- Split documents into chunks
- Create embeddings using Ollama
- Build two FAISS indexes:
  - `indexes/chunk_idx/` - For chunk-level retrieval
  - `indexes/doc_idx/` - For document-level routing

### Step 2: Start the API Server

```cmd
python -m app.main
```

Or with uvicorn directly:
```cmd
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The server will start at `http://127.0.0.1:8000`

### Step 3: Interact with the Chatbot

**Using the Interactive API Docs:**
1. Open your browser to `http://127.0.0.1:8000/docs`
2. Try the `/ask` endpoint with a question

**Using curl:**
```cmd
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is semantic data modeling?\"}"
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "What is semantic data modeling?"}
)
print(response.json())
```

### API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint
- `POST /ask` - Ask a question
  - Request body: `{"question": "your question here"}`
  - Response includes:
    - `answer`: The generated answer
    - `sources`: List of source documents with page numbers
    - `snippets`: Relevant text excerpts from sources
    - `retrieved_count`: Number of chunks retrieved
    - `used_count`: Number of chunks actually used

## How It Works

### Two-Stage Retrieval

1. **Document Routing**: First, the system identifies the most relevant documents (top-N) using the doc routing index
2. **Chunk Retrieval**: Then, it retrieves the most relevant chunks only from those documents using the chunk index
3. **Answer Generation**: The LLM generates an answer based on the retrieved chunks with proper citations

This approach improves precision and reduces noise from irrelevant documents.

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TOP_K` | 4 | Number of chunks to use for answer generation |
| `CHUNK_SIZE` | 800 | Size of each text chunk in characters |
| `CHUNK_OVERLAP` | 120 | Overlap between consecutive chunks |
| `DOC_ROUTE_TOP_N` | 3 | Number of documents to route to |
| `CHUNK_FETCH_K` | 40 | Number of chunks to retrieve before filtering |
| `DOC_TEXT_MAX_CHARS` | 12000 | Maximum characters per document for routing |

## Evaluation

Run retrieval evaluation:
```cmd
python -m eval.eval_retrieval
```

Run answer quality evaluation:
```cmd
python -m eval.eval_answers
```

## Troubleshooting

**Ollama not running:**
- Make sure Ollama is installed and running: `ollama serve`
- Check if models are available: `ollama list`

**Index not found:**
- Run the ingestion script first: `python -m ingests.ingest`
- Ensure PDF files exist in `data/pdfs/`

**Slow responses:**
- Use smaller models (e.g., `phi3:mini` instead of larger models)
- Reduce `RAG_TOP_K` and `CHUNK_FETCH_K` values
- Consider using GPU acceleration with Ollama

**Empty or poor answers:**
- Verify PDFs contain relevant information
- Adjust chunk size and overlap parameters
- Increase `RAG_TOP_K` to provide more context

## Limitations

- Embedding quality depends on the nomic-embed-text model (good for demos, may need fine-tuning for production)
- Citation accuracy is enforced via prompts but not guaranteed
- Performance depends on local hardware (CPU/GPU)
- Currently only supports Ollama as LLM provider

## License

Both Ollama models (chat model and embeddings model) llama3.2:3b and nomic-embed-text are open source




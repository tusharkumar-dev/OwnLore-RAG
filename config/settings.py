import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# reranker
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N  = 5
RERANKER_ENABLE = True

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

_MODEL_MAP = {
    "groq":        "llama-3.1-8b-instant",
    "openai":      "gpt-4o",
    "ollama":      "llama3.1:latest",
    "gemini":      "gemini-3.5-flash",
    "anthropic":   "claude-sonnet-4-5",       
    "openrouter":  "mistralai/mistral-7b-instruct",
    "deepseek":    "deepseek-chat",
    "grok":        "grok-3",    
}

MODEL        = _MODEL_MAP.get(LLM_PROVIDER, "openai/gpt-oss-120b")
ACTIVE_MODEL = os.environ.get("ACTIVE_MODEL", MODEL)

LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_STREAMING   = os.environ.get("LLM_STREAMING", "true").lower() == "true"
LLM_MAX_TOKENS  = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

DB_PATH         = os.environ.get("CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

SUMMARY_MAX_CHUNKS = 15
SUMMARY_MAX_CHARS  = 300 

TOP_K           = int(os.environ.get("TOP_K", "5"))
SEARCH_TYPE     = os.environ.get("SEARCH_TYPE", "similarity")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.5"))

CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))


# Validate configuration at startup.

_VALID_PROVIDERS = {
    "groq", "openai", "ollama", "gemini",
    "anthropic", "openrouter", "deepseek",
    "grok",
}

_VALID_SEARCH_TYPES = {"similarity", "mmr", "similarity_score_threshold"}

if LLM_PROVIDER not in _VALID_PROVIDERS:
    raise ValueError(f"Invalid LLM_PROVIDER='{LLM_PROVIDER}'. Valid: {_VALID_PROVIDERS}")

if SEARCH_TYPE not in _VALID_SEARCH_TYPES:
    raise ValueError(f"Invalid SEARCH_TYPE='{SEARCH_TYPE}'. Valid: {_VALID_SEARCH_TYPES}")

if not (1 <= TOP_K <= 20):
    raise ValueError(f"TOP_K={TOP_K} out of range. Must be 1-20.")

if not (0.0 <= LLM_TEMPERATURE <= 2.0):
    raise ValueError(f"LLM_TEMPERATURE={LLM_TEMPERATURE} out of range. Must be 0.0-2.0.")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(f"CHUNK_OVERLAP={CHUNK_OVERLAP} must be < CHUNK_SIZE={CHUNK_SIZE}.")
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq as NativeGroqClient

from ..exceptions import ConfigError

# ==============================================================================
# # Paths
# ==============================================================================


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
MODEL_CACHE_DIR = ROOT_DIR / ".model_cache"
QDRANT_STORAGE_PATH = DATA_DIR / "qdrant_db"

load_dotenv(ROOT_DIR / ".env")

model_cache_path = ROOT_DIR / ".model_cache"
model_cache_path.mkdir(exist_ok=True)


# ==============================================================================
# Embedding Model Constants & Collection name
# ==============================================================================


DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DEFAULT_COLLECTION_NAME = "textbooks"


# ==============================================================================
# Configure Gemini for Vision2
# ==============================================================================


GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.5-flash-lite:generateContent"
)


def get_gemini_api_key() -> str:
    """Retrieves GEMINI_API_KEY or raises ConfigError."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Visual page extraction requires Gemini. "
            "Please add GEMINI_API_KEY to your .env file."
        )
    return api_key.strip()


# ==============================================================================
# Configure Groq
# ==============================================================================


GROQ_MAIN_MODEL = "openai/gpt-oss-120b"


def get_groq_api_key() -> str:
    """Retrieves GROQ_API_KEY or raises ConfigError."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ConfigError(
            "GROQ_API_KEY is not set. Please add GROQ_API_KEY to your .env file."
        )
    return api_key.strip()


def get_groq_client() -> NativeGroqClient:
    """Returns an authenticated native Groq client with timeout and retries."""
    return NativeGroqClient(api_key=get_groq_api_key(), timeout=30.0, max_retries=3)

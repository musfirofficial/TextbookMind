import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq as NativeGroqClient
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq as LlamaGroq

# Project root is 2 levels up from src/rag/
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

model_cache_path = ROOT_DIR / ".model_cache"
model_cache_path.mkdir(exist_ok=True)


from rag.exceptions import ConfigError

# Config Gemini
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


# Config Groq Models
GROQ_TOC_MODEL = "qwen/qwen3.8-27b"
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
    """Returns an authenticated native Groq client."""
    return NativeGroqClient(api_key=get_groq_api_key())


# Config FastEmbed
def get_fastembed_config():
    Settings.embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        max_length=512,
        threads=4,
        cache_dir=str(model_cache_path),
    )


# Initialize LlamaIndex with Groq LLM and FastEmbed
def init_llama_index():

    # 1. Configure Groq LLM
    Settings.llm = LlamaGroq(
        model=GROQ_MAIN_MODEL,
        api_key=get_groq_api_key(),
        temperature=0.2,
    )

    # 2. Configure FastEmbed
    Settings.embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        max_length=512,
        threads=4,
        cache_dir=str(model_cache_path),
    )

    print(f"Global LlamaIndex Settings Initialized. Cache: {model_cache_path}")


# Condense Config
def get_condense_llm():
    return LlamaGroq(
        model="qwen/qwen3.8-27b",
        api_key=get_groq_api_key(),
        temperature=0.1,  # Low temperature for strict factual rewriting
    )


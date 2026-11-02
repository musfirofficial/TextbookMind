import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq

# Project root is 2 levels up from src/rag/
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def init_llama_index():
    model_cache_path = ROOT_DIR / ".model_cache"
    model_cache_path.mkdir(exist_ok=True)

    # 1. Configure Groq LLM
    Settings.llm = Groq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
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

import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq

load_dotenv()


def init_llama_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_cache_path = os.path.join(base_dir, ".model_cache")
    os.makedirs(model_cache_path, exist_ok=True)

    # 1. Configure Groq LLM
    Settings.llm = Groq(
        model="openai/gpt-oss-120b",  # llama-3.3-70b-versatile Or openai/gpt-oss-120b
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    # 2. Configure FastEmbed (Notice: Settings.embed_model)
    Settings.embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        max_length=512,
        threads=4,
        cache_dir=model_cache_path,
    )

    print(f"Global LlamaIndex Settings Initialized. Cache: {model_cache_path}")

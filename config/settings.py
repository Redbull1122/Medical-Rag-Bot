import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-rag-bot")
    # Ollama connection (inside Docker use host.docker.internal)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def validate_config(key: str):
    """
    Checks whether the specified key exists and has a value in Config.
    """
    value = getattr(Config, key, None)
    if not value:
        raise ValueError(f"Config key '{key}' is missing or empty!")
    return value


def validate_all_config():
    """
    Checks all mandatory configuration keys.
    """
    required_keys = ["PINECONE_API_KEY", "LANGFUSE_SECRET_KEY", "PINECONE_INDEX_NAME"]
    for key in required_keys:
        validate_config(key)
    print("All config keys are valid!")




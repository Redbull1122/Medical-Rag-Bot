# MedlinePlus Medical Assistant

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Pinecone, Langgraph, and Streamlit for querying **MedlinePlus** medical content. This project demonstrates LLMOps practices with containerization, monitoring, and scalable deployment.

## Features

- **RAG Pipeline**: Retrieves and ranks relevant medical documents from MedlinePlus using embeddings.
- **LangChain Integration**: Handles LLM interactions and orchestrates the query workflow.
- **Langgraph Integration**: Builds a flow of the application with memory and context management.
- **Streamlit Frontend**: User-friendly chat interface with continuous conversation history.
- **Monitoring**: Integrated with LangFuse for tracing, analytics, and conversation tracking.
- **Containerized**: Docker support for easy deployment.

## Project Structure

```
medical-rag-bot/
├── config/                         # Configuration files
│   ├── __init__.py                 # Ініціалізація модуля
│   └── settings.py                 # Project settings (API keys, URLs)
├── src/                            # Main program code
│   ├── api/                        # API integration
│   │   ├── __init__.py
│   │   └── medlineplus.py          # Integration with MedlinePlus API
│   ├── core/                       # Core app logic
│   │   ├── __init__.py
│   │   ├── langraph_workflow.py    # LangGraph workflow for RAG
│   │   ├── memory.py               # Memory and context management
│   │   └── vector_store.py         # Vector database (embeddings)
│   └── utils/                      # Supporting utilities
│       ├── __init__.py
│       └── text_processing.py      # Text processing and cleaning
├── .dockerignore                   # Docker build context ignore
├── .gitignore                      # Git ignore
├── Dockerfile                      # Docker configuration
├── main.py                         # Main application file (Streamlit)
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```



## Prerequisites

- Python 3.11
- Docker (for containerization)
- API keys for Pinecone, LangFuse

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/medical-rag-bot.git
   cd medical-rag-bot

## Set up environment:

- pip install -r requirements.txt


## Configure environment variables: Create a .env file in the root directory:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host
```

## Run locally

```bash
streamlit run main.py
```

## Docker

```bash
docker build -t medical-rag-bot .
docker run --rm -p 8501:8501 --env-file .env medical-rag-bot
```





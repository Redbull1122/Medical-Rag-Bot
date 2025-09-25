from src.api.medlineplus import search_medline
from src.utils.text_processing import normalize_text, extract_key_sentences
from src.core.vector_store import upsert_embeddings, search_embeddings, format_search_results
from config.settings import Config
from langfuse import Langfuse
from langfuse import observe as lf_observe
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import add_messages, StateGraph, END
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from src.core.memory import memory
from langchain.schema import Document
import uuid


def _noop_observe(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator

# Initialize Langfuse only if all config values are present and reachable
USE_LANGFUSE = bool(Config.LANGFUSE_SECRET_KEY and Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_HOST)
if USE_LANGFUSE:
    try:
        langfuse = Langfuse(
            secret_key=Config.LANGFUSE_SECRET_KEY,
            public_key=Config.LANGFUSE_PUBLIC_KEY,
            host=Config.LANGFUSE_HOST
        )
        observe = lf_observe
    except Exception:
        langfuse = None
        observe = _noop_observe
else:
    langfuse = None
    observe = _noop_observe

model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatOllama(model=Config.OLLAMA_MODEL, temperature=0.6, base_url=Config.OLLAMA_BASE_URL)

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {
       "messages": [llm.invoke(state["messages"])]
    }


def ingest_from_medline(term: str):
    try:
        docs = search_medline(term)
        processed_docs = []

        for i, doc in enumerate(docs):
            title = normalize_text(doc.get("title", "Без назви"))
            summary = normalize_text(doc.get("summary", ""))

            key_sentences = extract_key_sentences(summary, max_sentences=2)
            combined_text = f"{title}. {' '.join(key_sentences)}"

            # Generate embeddings
            vector = model.encode(combined_text).tolist()

            unique_id = str(uuid.uuid4())

            upsert_embeddings(
                [combined_text],
                [{"title": title, "summary": summary, "url": doc.get("url", "#")}],
                [vector],
                ids=[unique_id]
            )

            processed_docs.append(
                Document(
                    page_content=combined_text,
                    metadata={"title": title, "summary": summary, "url": doc.get("url", "#")}
                )
            )

        print("Debug: ingest_from_medline returned", processed_docs)
        return processed_docs

    except Exception as e:
        print(f"Error in ingest_from_medline: {e}")

@observe(name="llm-call", as_type="generation")
async def query_pipeline(user_query: str, thread_id: str = "default_conversation"):
    """
    Search for relevant documents in Pinecone
    Use LLM with context
    Langgraph automatically manages conversation history via checkpointer
    """
    try:
        # Searching for relevant documents
        results = search_embeddings(user_query, top_k=3)
        context = format_search_results(results) if results.get("matches") else "No relevant documents found."

        # Creating messages with context
        messages = [
            SystemMessage(content="You are a helpful medical assistant. Answer questions based on the provided context and conversation history."),
            HumanMessage(content=f"Context from medical database:\n{context}\n\nQuestion: {user_query}")
        ]

        # Creating a graph
        graph = StateGraph(BasicChatState)
        graph.add_node("chatbot", chatbot)
        graph.add_edge("chatbot", END)
        graph.set_entry_point("chatbot")

        # Compiling a graph with checkpointer (automatically manages history)
        app = graph.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": thread_id}}
        
        # Call the graph - langgraph will automatically add a new message to the history
        result = app.invoke({"messages": messages}, config=config)
        
        return result["messages"][-1].content
        
    except Exception as e:
        print(f"Error in query_pipeline: {e}")
        # Provide a more actionable message for connection issues
        msg = str(e)
        if "Connection refused" in msg or "Errno 111" in msg:
            return (
                "Connection error. Check that external services are reachable: "
                "- OLLAMA_BASE_URL (Ollama), - PINECONE_API_KEY/index, - LANGFUSE_HOST (if enabled)."
            )
        return f"Sorry, an error occurred while processing your request: {e}"

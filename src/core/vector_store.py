from config.settings import Config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import hashlib

#Initialize the Pinecone client with the API key
pc = Pinecone(api_key=Config.PINECONE_API_KEY)

#Downloading the model for building embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

index_name = Config.PINECONE_INDEX_NAME
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "all-MiniLM-L6-v2",
            "field_map": {"text": "chunk_text"}
        }
    )

dense_index = pc.Index(index_name)



def upsert_embeddings(texts, metadata_list=None, vectors=None, ids=None):
    """
    Adds texts to Pinecone with corresponding metadata.
    texts: list of texts
    metadata_list: list of dictionaries with metadata (can be None)
    vectors: list of pre-computed vectors (optional, if not provided will generate them)
    upsert result
    """
    records = []
    for i, text in enumerate(texts):
        # Use the ready vector if provided, otherwise generate it.
        if vectors and i < len(vectors):
            vector = vectors[i]
        else:
            vector = model.encode(text).tolist()

        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {"text": text}
        # Determine a stable id: prefer provided ids, otherwise hash of text
        if ids and i < len(ids) and ids[i]:
            record_id = str(ids[i])
        else:
            record_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        records.append({
            "id": record_id,
            "values": vector,
            "metadata": metadata
        })
    return dense_index.upsert(namespace="medical-bot", vectors=records)

def search_embeddings(query, top_k=5):
    """
    Performs a search for relevant documents in Pinecone.
    query: user query (text)
    top_k: number of most relevant results
    search results with metadata
    """
    vector = model.encode(query).tolist()
    results = dense_index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results

def format_search_results(results):
    """
    Formats search results into a format suitable for LLM.
    results: results from Pinecone
    formatted text context
    """
    context_texts = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        score = match.get("score", 0.0)
        title = metadata.get("title") or metadata.get("text", "Без назви")
        summary = metadata.get("summary") or metadata.get("text", "")
        context_texts.append(f"{title}: {summary} (score: {score:.2f})")
    return "\n\n".join(context_texts)







import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")

def embed(text: str) -> list[float]:
    #this exists here as text2vec-ollama module ignores the collection's apiEndpoint on hybrid path
    #open issue Weaviate #8406
    response = httpx.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "input": text})
    response.raise_for_status()
    return response.json()["embeddings"][0]

def search_db(collection, text: str):
    query_vector = embed(text)

    result = collection.query.hybrid(
        query=text,
        vector=query_vector,
        alpha=0.8,
        limit=8,
    )
    return [
        {
            "url": obj.properties["url"],
            "heading": obj.properties["heading"],
            "chunk": obj.properties["chunk"],
        }
        for obj in result.objects
    ]

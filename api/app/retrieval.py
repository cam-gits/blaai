import os
import httpx
from weaviate.classes.query import MetadataQuery, HybridFusion
from app.config import ALPHA, LIMIT, TEMPERATURE, TOP_P, MAX_TOKENS, NUM_CTX, SEED, DISTANCE_CUTOFF

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
GEN_MODEL = os.getenv("LLM_MODEL", "phi4-mini:3.8b-q4_K_M")

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
        alpha=ALPHA,
        limit=LIMIT,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        return_metadata=MetadataQuery(score=True),
    )
    chunks = [
        {
            "url": obj.properties["url"],
            "heading": obj.properties["heading"],
            "chunk": obj.properties["chunk"],
            "score": obj.metadata.score,
        }
        for obj in result.objects
    ]

    gate = collection.query.near_vector(
        near_vector=query_vector,
        limit=1,
        return_metadata=MetadataQuery(distance=True),
    )
    top_distance = gate.objects[0].metadata.distance if gate.objects else None

    return chunks, top_distance

def generate(
    prompt: str,
    role: str = "user",
    model: str = GEN_MODEL,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    num_predict: int = MAX_TOKENS,
    num_ctx: int = NUM_CTX,
    seed: int | None = SEED,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": role, "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    resp = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["message"]["content"]

def prompt(collection, query: str):
    chunks, top_distance = search_db(collection, query)

    if not (chunks and top_distance is not None and top_distance < DISTANCE_CUTOFF):
        return "I don't have that information in my sources.", []

    sources = "\n\n".join(
    f"[Source {i+1}] {c['heading']}\nURL: {c['url']}\n{c['chunk']}"
    for i, c in enumerate(chunks)
)

    prompt = f"""You are Blaa AI, an assistant that answers questions about local government services in Waterford, Ireland.

                You answer ONLY from the SOURCES provided below. Follow these rules:

                1. Use only facts stated in the SOURCES. Do not add information from your own knowledge.
                2. Never invent or guess contact details. Only give a phone number, email address, office location, or name of an official if it appears verbatim in the SOURCES.
                3. If the SOURCES do not contain the answer, tell the user you don't have that information in your sources. Then suggest the user check the relevant council website or contact the council directly. Do not attempt a partial or approximate answer.
                4. Only answer questions about Waterford local government services, or Waterford CIty generally. If asked about anything else, briefly say it's outside what you cover.
                5. Do not specify source number or URL in your answer, these will be added seperately. Do not use phrases such as "According to Source 6" or "as mentioned in Source 3", or even include "(Source 2)" after claims - instead say "According to Waterford County Council" or "According to the Planning Department"

                SOURCES:
                {sources}

                QUERY:
                {query}"""

    result = generate(prompt)

    seen = set()
    urls = [c["url"] for c in chunks if not (c["url"] in seen or seen.add(c["url"]))]

    return result, urls


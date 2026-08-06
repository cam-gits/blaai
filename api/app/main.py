import os
import json
import httpx
import weaviate
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import retrieval

OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
WEAVIATE_HTTP_PORT = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
WEAVIATE_HOST = os.environ["WEAVIATE_HOST"]
WEAVIATE_URL = f"http://{WEAVIATE_HOST}:8080"
LLM_MODEL = os.environ["LLM_MODEL"]

COLLECTION_NAME = "blaai_collection"

log = logging.getLogger("blaai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    #Open one Weaviate connection for the life of the process
    app.state.client = weaviate.connect_to_local(host=WEAVIATE_HOST)
    app.state.collection = app.state.client.collections.get(COLLECTION_NAME)
    try:
        yield
    finally:
        app.state.client.close()


app = FastAPI(title="Blaai API", lifespan=lifespan)

class Query(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/deps")
async def health_deps():
    #Confirm the API can reach its dependencies
    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        #Weaviate readiness
        try:
            r = await client.get(f"{WEAVIATE_URL}/v1/.well-known/ready")
            results["weaviate"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except Exception as e:
            results["weaviate"] = f"unreachable: {e.__class__.__name__}"

        #Ollama tags endpoint
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            results["ollama"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except Exception as e:
            results["ollama"] = f"unreachable: {e.__class__.__name__}"

    return results


def _line(obj) -> str:
    #one JSON object per line - NDJSON
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.post("/ask")
def ask(query: Query):
    #Retrieval runs before the response starts, so failures here can still
    #return a real status code. Once the first byte is sent, the status is
    #committed and problems have to be reported inside the stream instead.
    try:
        prompt_text, urls = retrieval.build_context(app.state.collection, query.question)
    except httpx.TimeoutException:
        log.exception("upstream timeout during retrieval")
        raise HTTPException(504, "Search took too long. Please try again.")
    except httpx.HTTPError:
        log.exception("upstream failure during retrieval")
        raise HTTPException(502, "Search is unavailable right now.")
    except Exception:
        log.exception("unhandled error during retrieval")
        raise HTTPException(500, "Something went wrong handling that question.")

    def stream():
        #Refused by the distance gate - answer without troubling the model
        if prompt_text is None:
            yield _line({"type": "token", "text": retrieval.REFUSAL})
            yield _line({"type": "done", "sources": []})
            return

        try:
            for token in retrieval.generate_stream(prompt_text):
                yield _line({"type": "token", "text": token})
        except Exception:
            log.exception("generation failed mid-stream")
            yield _line({"type": "error", "message": "The answer was cut short. Please try again."})
            return

        yield _line({"type": "done", "sources": urls})

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
    )
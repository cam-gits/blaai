from weaviate.classes.config import Configure, Property, DataType
import weaviate
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_ENDPOINT = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")

with weaviate.connect_to_local(host=WEAVIATE_HOST) as client:
    if client.collections.exists('blaai_collection'):
        print("Collection already exists, nothing to do.")
    else:
        client.collections.create(
            name='blaai_collection',
            vectorizer_config=Configure.Vectorizer.text2vec_ollama(
                api_endpoint=OLLAMA_ENDPOINT,
                model=EMBED_MODEL,
            ),
            properties=[
                Property(name='url', data_type=DataType.TEXT, skip_vectorization=True),
                Property(name='heading', data_type=DataType.TEXT),
                Property(name='chunk', data_type=DataType.TEXT),
                Property(name='chunk_index', data_type=DataType.INT, skip_vectorization=True),
            ],
        )
        print("Collection created.")
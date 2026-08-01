from weaviate.classes.config import Configure, Property, DataType
import weaviate
from weaviate.util import generate_uuid5
import json
from tqdm import tqdm
from pprint import pprint
import os
from dotenv import load_dotenv
load_dotenv()

client = weaviate.connect_to_local()

OLLAMA_ENDPOINT = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
IN_PATH = "data/raw/chunks.jsonl"

if client.collections.exists('blaai_collection'):
    client.collections.delete('blaai_collection')

collection = client.collections.create(
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

with open(IN_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

with collection.batch.fixed_size(batch_size=100, concurrent_requests=1) as batch:
    # Iterate over a subset of the dataset
    for line in tqdm(lines):
            document = json.loads(line)

            uuid = generate_uuid5(f"{document['URL']}_{document['chunk_index']}")

            batch.add_object(
                properties={
                    "url": document["URL"],
                    "heading": document["heading"],
                    "chunk": document["chunk"],
                    "chunk_index": int(document["chunk_index"]),
                },
                uuid=uuid,
            )


if collection.batch.failed_objects:
    print(f"Failed: {len(collection.batch.failed_objects)}")
    pprint(collection.batch.failed_objects[:3])
else:
    print(f"Inserted {len(collection)} objects")

client.close()
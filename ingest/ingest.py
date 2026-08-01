import weaviate
from weaviate.util import generate_uuid5
import json
from tqdm import tqdm
from pprint import pprint
import os
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
IN_PATH = "data/raw/chunks.jsonl"

with weaviate.connect_to_local(host=WEAVIATE_HOST) as client:
    if not client.collections.exists('blaai_collection'):
        raise SystemExit("Collection doesn't exist. Run schema.py first.")

    collection = client.collections.get('blaai_collection')

    with open(IN_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with collection.batch.fixed_size(batch_size=100, concurrent_requests=1) as batch:
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
        print(f"Inserted/updated {len(collection)} objects")
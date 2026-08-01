import weaviate
import os
from dotenv import load_dotenv

load_dotenv()
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")

with weaviate.connect_to_local(host=WEAVIATE_HOST) as client:
    if client.collections.exists('blaai_collection'):
        client.collections.delete('blaai_collection')
        print("Collection deleted.")
    else:
        print("Nothing to delete.")
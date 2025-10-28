from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import subprocess
import time
import requests

class DatabaseClient:
    def __init__(self, vector_size: int):
        self.client = QdrantClient("http://localhost:6333")
        self.vector_size = vector_size
        self.db_process = None

    def open_connection(self, timeout: int = 60):
        # Run Qdrant in a Docker container
        self.db_process = subprocess.Popen(
            ["docker", "run", "--rm", "--name", "qdrant_db", "-p", "6333:6333", "qdrant/qdrant"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("YES THIS IN NEW VERSION")
        # Wait for Qdrant to be ready
        start = time.time()
        while True:
            try:
                response = requests.get("http://localhost:6333/readyz")
                if response.status_code == 200:
                    return
            except requests.ConnectionError:
                pass

            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for Qdrant to start.")
            time.sleep(1)

        
    def close_connection(self):
        if self.db_process:
            subprocess.run(["docker", "stop", "qdrant_db"])
            self.db_process.terminate()
            self.db_process.wait()
            time.sleep(2)  # Give some time for the process to terminate
        else:
            raise RuntimeError("No active database process to terminate.")

    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def upsert_vectors(self, collection_name: str, points: list[dict]):
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )

    def vector_search(self, collection_name: str, vector: list[float], top_k: int = 5):
        return self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k
        )
from src.basic_code_search.database_client import DatabaseClient
from src.basic_code_search.embedding_model import EmbeddingModel
from src.basic_code_search.metrics import RecallAt10, MRRAt10, NDCGAt10
from qdrant_client.models import PointStruct
import uuid

class SearchEngine:
    def __init__(self, embedding_model: EmbeddingModel, collection_name: str = 'documents', top_k: int = 5):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.top_k = top_k

        self.db_client = DatabaseClient(embedding_model.get_model().get_sentence_embedding_dimension())

    def open(self):
        self.db_client.open_connection()
        self.db_client.create_collection(self.collection_name)

    def close(self):
        self.db_client.close_connection()

    def load_text_data(self, texts: list[str], ids: list[str] | None = None, batch_size: int = 1024, verbose: bool = False):
        if verbose:
            print(f"Encoding {len(texts)} texts into embeddings...")
        embeddings = self.embedding_model.encode(texts)
        if verbose:
            print(f"Encoding done.\n")

        if ids is None:
            id_list = [None] * len(texts)
        else:
            id_list = ids

        if verbose:
            print(f"Upserting {len(texts)} vectors into the database in batches of {batch_size}...")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"dataset_id": id, "text": text}
                ) for id, emb, text in zip(id_list, embeddings, texts)
        ]

        for i, start in enumerate(range(0, len(points), batch_size)):
            batch_points = points[start:start + batch_size]
            self.db_client.upsert_vectors(self.collection_name, batch_points)
            if verbose:
                print(f"Upserted batch {i + 1} with {len(batch_points)} vectors.")

    def search(self, query: str):
        query_embedding = self.embedding_model.encode(query)
        results = self.db_client.vector_search(self.collection_name, query_embedding, top_k=self.top_k)
        return results
        
    def evaluate(self, predictions: list[list], targets: list):
        recall_at_10 = RecallAt10(predictions, targets)
        mrr_at_10 = MRRAt10(predictions, targets)
        ndcg_at_10 = NDCGAt10(predictions, targets)
        
        print("Evaluation Metrics:")
        print(f"Recall@10: {recall_at_10:.4f}")
        print(f"MRR@10: {mrr_at_10:.4f}")
        print(f"NDCG@10: {ndcg_at_10:.4f}")
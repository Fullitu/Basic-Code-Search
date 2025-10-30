from src.basic_code_search.database_client import DatabaseClient
from src.basic_code_search.embedding_model import EmbeddingModel
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

    def load_text_data(self, texts: list[str], ids: list[str] | None = None):
        embeddings = self.embedding_model.encode(texts)
        
        if ids is None:
            id_list = [None] * len(texts)
        else:
            id_list = ids

        points = [
            PointStruct(
                id=id if id else str(uuid.uuid4()),
                vector=emb,
                payload={"text": text}
                ) for id, emb, text in zip(id_list, embeddings, texts)
        ]

        self.db_client.upsert_vectors(self.collection_name, points)

    def search(self, query: str):
        query_embedding = self.embedding_model.encode(query)
        results = self.db_client.vector_search(self.collection_name, query_embedding, top_k=5)
        return results
        
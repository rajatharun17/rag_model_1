from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from typing import List, Dict, Any
import time

class FastVectorStore:
    def __init__(self, collection_name: str = "tamil_msmarco"):
        # In-Memory Qdrant for sub-10ms retrieval speeds
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        
        # FastEmbed uses optimized ONNX runtime
        self.encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.vector_size = 384
        
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 256):
        texts = [c["text"] for c in chunks]
        embeddings = list(self.encoder.embed(texts, batch_size=batch_size))
        
        points = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload=chunk
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        query_vector = list(self.encoder.embed([query]))[0].tolist()
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        retrieval_ms = (time.perf_counter() - t0) * 1000
        
        results = []
        for res in search_results:
            results.append({
                "score": res.score,
                "payload": res.payload,
                "retrieval_ms": retrieval_ms
            })
        return results
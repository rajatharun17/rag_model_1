import re
from typing import List, Dict, Any

class AdvancedChunker:
    """Supports Fixed-Window, Semantic Sentence, and Hybrid Chunking."""
    
    @staticmethod
    def fixed_size_chunking(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += (chunk_size - overlap)
        return chunks

    @staticmethod
    def semantic_sentence_chunking(text: str, max_words_per_chunk: int = 200) -> List[str]:
        # Split Tamil/English text by sentence boundaries (. | ? ! \n)
        sentences = re.split(r'(?<=[.!?|])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_length + sentence_word_count > max_words_per_chunk and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_length += sentence_word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    @classmethod
    def process_record(cls, record: Dict[str, Any], strategy: str = "hybrid") -> List[Dict[str, Any]]:
        doc_id = record.get("id", "unknown")
        doc_title = record.get("title", "")
        body_text = record.get("passage", record.get("text", record.get("body", "")))

        if strategy == "fixed":
            raw_chunks = cls.fixed_size_chunking(body_text)
        elif strategy == "semantic":
            raw_chunks = cls.semantic_sentence_chunking(body_text)
        else:  # Metadata-Aware Hybrid Strategy
            raw_chunks = cls.semantic_sentence_chunking(f"{doc_title}. {body_text}")

        chunk_docs = []
        for idx, chunk in enumerate(raw_chunks):
            chunk_docs.append({
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "title": doc_title,
                "text": chunk,
                "strategy": strategy
            })
        return chunk_docs
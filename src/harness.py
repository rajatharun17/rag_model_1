import time
from typing import Dict, Any, List
import re

class SafetyGuardrail:
    UNSAFE_KEYWORDS = ["hack", "exploit", "attack", "malware", "violence"]
    
    @classmethod
    def validate_input(cls, query: str) -> bool:
        if not query:
            return False
            
        # Strip square brackets / tags added by Speech-to-Text
        cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', query).strip()
        
        # Accept any query with at least 2 non-tag characters
        if len(cleaned) < 2:
            return False

        # Reject explicit unsafe keywords
        for word in cls.UNSAFE_KEYWORDS:
            if word in cleaned.lower():
                return False
                
        return True
    @classmethod
    def check_groundedness(cls, retrieved_results: List[Dict[str, Any]], threshold: float = 0.35) -> bool:
        if not retrieved_results:
            return False
        top_score = retrieved_results[0]["score"]
        return top_score >= threshold

class OrchestrationHarness:
    def __init__(self, vector_store, max_retries: int = 2):
        self.vector_store = vector_store
        self.max_retries = max_retries

    def run_pipeline(self, query_text: str) -> Dict[str, Any]:
        pipeline_start = time.perf_counter()
        
        # Guardrail 1: Input Validation
        if not SafetyGuardrail.validate_input(query_text):
            return {
                "status": "REJECTED",
                "reason": "Off-topic or safety violation.",
                "answer": "I cannot answer this query due to safety or input policy.",
                "latency_ms": (time.perf_counter() - pipeline_start) * 1000
            }

        # Vector Retrieval with retry wrapper
        retries = 0
        results = []
        while retries <= self.max_retries:
            try:
                results = self.vector_store.search(query_text, top_k=3)
                break
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    return {
                        "status": "ERROR",
                        "reason": f"Retrieval failed after {self.max_retries} retries: {str(e)}",
                        "latency_ms": (time.perf_counter() - pipeline_start) * 1000
                    }

        # Guardrail 2: Context Grounding / Hallucination prevention
        if not SafetyGuardrail.check_groundedness(results):
            return {
                "status": "UNGROUNDED",
                "reason": "Retrieved context confidence is below threshold.",
                "answer": "I do not have sufficient grounded context from the dataset to answer this.",
                "latency_ms": (time.perf_counter() - pipeline_start) * 1000
            }

        # Construct Answer
        context_str = "\n".join([r["payload"]["text"] for r in results])
        grounded_answer = f"Based on dataset context: {results[0]['payload']['text']}"

        total_latency = (time.perf_counter() - pipeline_start) * 1000
        
        return {
            "status": "SUCCESS",
            "query": query_text,
            "answer": grounded_answer,
            "context": context_str,
            "retrieval_ms": results[0]["retrieval_ms"],
            "total_latency_ms": total_latency
        }
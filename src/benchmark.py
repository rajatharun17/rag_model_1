import numpy as np
from typing import List, Dict, Any

class LatencyBenchmarker:
    @staticmethod
    def run_benchmark(harness, test_queries: List[str]) -> Dict[str, float]:
        latencies = []
        
        # Warmup query
        harness.run_pipeline("warmup query")
        
        for q in test_queries:
            res = harness.run_pipeline(q)
            latencies.append(res["total_latency_ms"])
            
        p50 = float(np.percentile(latencies, 50))
        p70 = float(np.percentile(latencies, 70))
        p100 = float(np.percentile(latencies, 100))
        
        return {
            "P50_ms": round(p50, 2),
            "P70_ms": round(p70, 2),
            "P100_ms": round(p100, 2),
            "samples": len(test_queries)
        }
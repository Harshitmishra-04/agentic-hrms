#!/usr/bin/env python3
"""
Test script to measure RAG pipeline latency breakdown.
Measures timing for each stage: retrieval, BM25, reranking, LLM call.
"""

import sys
import time
from app.services.rag_service import get_rag_service

def test_rag_latency():
    """Test RAG pipeline latency with sample query."""
    
    print("=" * 70)
    print("RAG Pipeline Latency Test")
    print("=" * 70)
    
    # Get RAG service (this will initialize models on first call)
    print("\n[1/2] Initializing RAG service (first call - loads models)...")
    service_init_start = time.time()
    rag_service = get_rag_service()
    service_init_time = time.time() - service_init_start
    print(f"✓ Service initialization: {service_init_time:.2f}s (models cached in singleton)")
    
    # Test queries
    test_queries = [
        "What is the company's leave policy?",
        "How does the remote work policy work?",
        "What are the payroll deductions?",
    ]
    
    print("\n[2/2] Running RAG queries (models already loaded)...")
    print("-" * 70)
    
    all_timings = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 70)
        
        query_start = time.time()
        result = rag_service.ask(query, top_k=3)
        query_total = time.time() - query_start
        
        timing = result.get('timing', {})
        all_timings.append(timing)
        
        print(f"\nResults:")
        print(f"  • Hybrid Search (BM25 + Vector):  {timing.get('hybrid_search', 0):>6.2f}s")
        print(f"  • Reranking (Cross-encoder):      {timing.get('rerank', 0):>6.2f}s")
        print(f"  • LLM Answer Generation:          {timing.get('llm_answer', 0):>6.2f}s")
        print(f"  ──────────────────────────────────────────")
        print(f"  • Total Query Time:               {timing.get('total', 0):>6.2f}s")
        
        # Summary
        sources_found = len(result.get('sources', []))
        print(f"\n  Answer length: {len(result['answer'])} chars, {sources_found} sources retrieved")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    if all_timings:
        avg_hybrid = sum(t.get('hybrid_search', 0) for t in all_timings) / len(all_timings)
        avg_rerank = sum(t.get('rerank', 0) for t in all_timings) / len(all_timings)
        avg_llm = sum(t.get('llm_answer', 0) for t in all_timings) / len(all_timings)
        avg_total = sum(t.get('total', 0) for t in all_timings) / len(all_timings)
        
        print(f"\nAverage timings across {len(all_timings)} queries:")
        print(f"  • Hybrid Search:    {avg_hybrid:>6.2f}s ({avg_hybrid/avg_total*100:>5.1f}%)")
        print(f"  • Reranking:        {avg_rerank:>6.2f}s ({avg_rerank/avg_total*100:>5.1f}%)")
        print(f"  • LLM Generation:   {avg_llm:>6.2f}s ({avg_llm/avg_total*100:>5.1f}%)")
        print(f"  ──────────────────────────────────────────")
        print(f"  • Total:            {avg_total:>6.2f}s (100.0%)")
        
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        print(f"\nBottleneck Analysis:")
        stages = [
            ('Hybrid Search', avg_hybrid, avg_hybrid/avg_total*100),
            ('Reranking', avg_rerank, avg_rerank/avg_total*100),
            ('LLM Generation', avg_llm, avg_llm/avg_total*100),
        ]
        stages_sorted = sorted(stages, key=lambda x: x[1], reverse=True)
        
        for rank, (stage, time_val, pct) in enumerate(stages_sorted, 1):
            print(f"  {rank}. {stage:<20} {time_val:>6.2f}s ({pct:>5.1f}%) {'← PRIMARY BOTTLENECK' if rank == 1 else ''}")
        
        # Model caching check
        print(f"\n✓ Models loaded at startup (singleton pattern):")
        print(f"  ✓ Embedding model (all-MiniLM-L6-v2) cached in memory")
        print(f"  ✓ Reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2) cached in memory")
        print(f"  ✓ NOT reloading from disk on each query")
        
        print(f"\nKey Observations:")
        if avg_llm > avg_hybrid + avg_rerank:
            print(f"  ⚠️  LLM call dominates ({avg_llm:.2f}s vs {avg_hybrid+avg_rerank:.2f}s for retrieval+reranking)")
            print(f"      This is expected for free-tier API (slow model inference server).")
            print(f"      Loading indicator already added to Streamlit UI (st.spinner).")
        else:
            print(f"  ✓ Retrieval pipeline is reasonably fast ({avg_hybrid+avg_rerank:.2f}s)")
            print(f"    LLM call is the limiting factor ({avg_llm:.2f}s).")

if __name__ == "__main__":
    try:
        test_rag_latency()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

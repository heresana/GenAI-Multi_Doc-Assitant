import time
from utils.vector_store import ChromaVectorStore

print("Initializing store (loading bi-encoder + cross-encoder)...")
t0 = time.time()
vs = ChromaVectorStore()
print(f"Init took {time.time()-t0:.2f}s")

# Fake chunks simulating two documents
doc_a_chunks = [
    {"content": "The vendor shall submit all invoices within 30 days of delivery.", "page": 1},
    {"content": "Late payment penalties are calculated at 1.5% per month on overdue invoices.", "page": 2},
    {"content": "The office is located near a park and has good weather most of the year.", "page": 3},
]
doc_b_chunks = [
    {"content": "Employees must complete compliance training annually by December 31st.", "page": 1},
    {"content": "Invoice payment terms for contractors are net-45 days from receipt.", "page": 2},
]

id_a = vs.index_document(doc_a_chunks, "contract.pdf")
id_b = vs.index_document(doc_b_chunks, "hr_policy.pdf")
print(f"Indexed doc_a={id_a}, doc_b={id_b}, total_chunks={vs.total_chunks()}")

query = "What is the invoice payment deadline?"

print("\n--- WITHOUT reranker (pure bi-encoder) ---")
t0 = time.time()
hits_no_rerank = vs.retrieve(query, top_k=3, use_reranker=False)
print(f"Latency: {time.time()-t0:.3f}s")
for h in hits_no_rerank:
    print(f"  dist={h['distance']:.4f} | {h['doc_name']} p{h['page']} | {h['content'][:60]}")

print("\n--- WITH cross-encoder reranker ---")
t0 = time.time()
hits_reranked = vs.retrieve(query, top_k=3, retrieve_k=5, use_reranker=True)
print(f"Latency: {time.time()-t0:.3f}s")
for h in hits_reranked:
    print(f"  rerank_score={h['rerank_score']:.4f} | {h['doc_name']} p{h['page']} | {h['content'][:60]}")

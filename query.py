"""
Text‑to‑video‑clip retrieval demo.
Run:
    python query.py --query "A car turns right at an intersection"
"""

import argparse, faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer

def search(index_path, meta_path, text, topk=5, model_name="all-MiniLM-L6-v2"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    model = SentenceTransformer(model_name)
    qvec  = model.encode([text], normalize_embeddings=True).astype("float32")
    D, I  = index.search(qvec, topk)

    print(f"\n🔎  Query: {text}\n")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        m = meta[int(idx)]
        print("\n================================================")
        print(f"{rank:>2}. score={score:.3f}  clip={m['timestamp']}  →  {m['caption'][:1000]}…")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--query", required=True)
    a.add_argument("--index", default="faiss.index")
    a.add_argument("--meta",  default="faiss_meta.pkl")
    a.add_argument("--model", default="all-MiniLM-L6-v2")
    a.add_argument("--topk",  type=int, default=5)
    args = a.parse_args()
    search(args.index, args.meta, args.query, args.topk, args.model)
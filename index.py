"""
Build a vector index from caption JSON produced by process_clips().
Run:
    python index.py --json data/4.json --index faiss.index
"""

import argparse, json, pickle, os, numpy as np, faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["captions"]           # list[dict(timestamp, caption)]

def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch=64):
    model = SentenceTransformer(model_name)
    vecs = []
    for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
        vecs.extend(model.encode(texts[i:i+batch], normalize_embeddings=True))
    return np.vstack(vecs).astype("float32")

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)     # cosine similarity (after L2‑norm)
    index.add(vectors)
    return index

def main(args):
    captions = load_json(args.json)

    sentences = [c["caption"] for c in captions]
    vectors   = embed_texts(sentences, args.model)

    # build & write index
    index = build_faiss_index(vectors)
    faiss.write_index(index, args.index)

    # save metadata (timestamp → caption)
    meta = {i: {"timestamp": captions[i]["timestamp"],
                "caption": captions[i]["caption"]}
            for i in range(len(captions))}
    with open(args.meta, "wb") as f:
        pickle.dump(meta, f)

    print(f"Indexed {len(captions)} captions -> {args.index}")
    print(f"Metadata saved         -> {args.meta}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json",  required=True)
    p.add_argument("--index", default="faiss.index")
    p.add_argument("--meta",  default="faiss_meta.pkl")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    main(p.parse_args())
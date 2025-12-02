import pandas as pd
from rank_bm25 import BM25Okapi
from preprocess import preprocess

def load_documents(path):
    df = pd.read_csv(path, encoding="latin1")
    docs = df["Article"].astype(str).tolist()
    ids = [f"doc_{i}" for i in range(len(docs))]
    return docs, ids

def build_bm25(docs):
    tokenized = [preprocess(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def search_bm25(bm25, docs, ids, query, k):
    q = preprocess(query)
    scores = bm25.get_scores(q)
    ranked = scores.argsort()[::-1][:k]
    results = []
    for idx in ranked:
        results.append((ids[idx], scores[idx], docs[idx][:200]))
    return results

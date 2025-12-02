from bm25_retrieval import load_documents, build_bm25, search_bm25
from tfidf_retrieval import build_tfidf, search_tfidf
from combined_retrieval import combine_results

DATA_PATH = "data/Articles.csv"

docs, ids = load_documents(DATA_PATH)

bm25, tokenized = build_bm25(docs)
tfidf_vec, tfidf_mat = build_tfidf(docs)

queries = [
    "petrol price",
    "cricket match sports",
    "stock market crash",
    "government fuel policy",
    "budget economy pakistan",
    "international oil rates"
]

for q in queries:
    print("\nQuery:", q)

    bm25_res = search_bm25(bm25, docs, ids, q, 10)
    print("\nBM25 Top Result:", bm25_res[0])

    tfidf_res = search_tfidf(tfidf_vec, tfidf_mat, docs, ids, q, 10)
    print("TF-IDF Top Result:", tfidf_res[0])

    combined = combine_results(bm25_res, tfidf_res)
    print("Combined Top Result:", combined[0])

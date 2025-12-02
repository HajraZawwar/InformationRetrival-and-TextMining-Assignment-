import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf(docs):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix

def search_tfidf(vec, mat, docs, ids, query, k):
    q = vec.transform([query])
    scores = cosine_similarity(q, mat).flatten()
    ranked = scores.argsort()[::-1][:k]
    results = []
    for idx in ranked:
        results.append((ids[idx], scores[idx], docs[idx][:200]))
    return results

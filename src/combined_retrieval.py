def combine_results(bm25_res, tfidf_res, alpha=0.7):
    score_map = {}

    for doc, score, _ in bm25_res:
        score_map[doc] = score_map.get(doc, 0) + alpha * score

    for doc, score, _ in tfidf_res:
        score_map[doc] = score_map.get(doc, 0) + (1 - alpha) * score

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    output = []
    for doc, score in ranked[:10]:
        output.append((doc, score))
    return output

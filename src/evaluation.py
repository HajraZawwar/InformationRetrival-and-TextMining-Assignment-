def precision_at_k(results, relevant, k):
    retrieved = [r[0] for r in results[:k]]
    match = sum(1 for x in retrieved if x in relevant)
    return match / k

def recall_at_k(results, relevant, k):
    retrieved = [r[0] for r in results[:k]]
    match = sum(1 for x in retrieved if x in relevant)
    return match / len(relevant) if len(relevant) else 0

def f1_score(p, r):
    return 2*p*r/(p+r) if (p+r)>0 else 0

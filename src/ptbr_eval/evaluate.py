import json
from pathlib import Path
from .bm25 import BM25
from .metrics import recall_at_k, reciprocal_rank


def load_corpus(path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r["text"]
    return out


def load_queries(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("relevant"):
                raise ValueError(f"query without relevant docs: {r.get('query')!r}")
            rows.append(r)
    return rows


def evaluate(corpus, queries, *, k=3, **bm25_opts) -> dict:
    idx = BM25(corpus, **bm25_opts)
    rec, rr, misses = [], [], []
    for q in queries:
        rel = set(q["relevant"])
        unknown = rel - set(corpus)
        if unknown:
            raise ValueError(f"unknown doc ids {sorted(unknown)} in query {q['query']!r}")
        ranked = [d for d, _ in idx.search(q["query"], k=len(corpus))]
        r = recall_at_k(ranked, rel, k)
        rec.append(r)
        rr.append(reciprocal_rank(ranked, rel))
        if r < 1.0:
            misses.append({"query": q["query"], "relevant": sorted(rel), "top": ranked[:k]})
    n = len(queries)
    return {"n": n, "k": k, f"recall@{k}": sum(rec) / n, "mrr": sum(rr) / n, "misses": misses}

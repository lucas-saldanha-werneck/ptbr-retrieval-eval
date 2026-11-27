"""Plain-Python BM25. No dependencies."""
import math
from collections import Counter
from .text import tokenize


class BM25:
    def __init__(self, docs: dict[str, str], *, fold_accents=True, drop_stopwords=True, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.opts = dict(fold_accents=fold_accents, drop_stopwords=drop_stopwords)
        self.ids = list(docs)
        self.tf = [Counter(tokenize(docs[i], **self.opts)) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = tokenize(query, **self.opts)
        scores = []
        for idx, doc_id in enumerate(self.ids):
            s = 0.0
            for t in q:
                f = self.tf[idx].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[idx] / (self.avg or 1))
                s += self.idf[t] * f * (self.k1 + 1) / denom
            scores.append((doc_id, s))
        # stable: score desc, then id asc
        scores.sort(key=lambda x: (-x[1], x[0]))
        return [(d, s) for d, s in scores[:k] if s > 0]

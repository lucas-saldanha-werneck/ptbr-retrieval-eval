import argparse
import json
import sys
from .evaluate import evaluate, load_corpus, load_queries


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ptbr-eval", description="Retrieval eval with a pass/fail gate.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--min-recall", type=float, default=None, help="exit 1 if Recall@k is below this")
    p.add_argument("--no-fold", action="store_true", help="disable accent folding (to compare)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    res = evaluate(load_corpus(a.corpus), load_queries(a.queries), k=a.k, fold_accents=not a.no_fold)
    key = f"recall@{a.k}"
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(f"queries={res['n']}  {key}={res[key]:.3f}  mrr={res['mrr']:.3f}")
        for m in res["misses"]:
            print(f"  MISS {m['query']!r}: want {m['relevant']}, got {m['top']}")
    if a.min_recall is not None and res[key] < a.min_recall:
        print(f"GATE FAILED: {key} {res[key]:.3f} < {a.min_recall}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

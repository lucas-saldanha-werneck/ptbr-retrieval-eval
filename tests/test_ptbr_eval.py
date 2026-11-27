import json
import pytest
from ptbr_eval.text import fold, tokenize
from ptbr_eval.bm25 import BM25
from ptbr_eval.metrics import recall_at_k, reciprocal_rank
from ptbr_eval.evaluate import evaluate, load_corpus, load_queries
from ptbr_eval.cli import main

CORPUS = "data/example_corpus.jsonl"
QUERIES = "data/example_queries.jsonl"


def test_fold_strips_accents():
    assert fold("Cartão Éter ÇÃ") == "cartao eter ca"


def test_tokenize_drops_stopwords():
    assert tokenize("o cartão de crédito") == ["cartao", "credito"]


def test_tokenize_without_fold_keeps_accents():
    assert "cartão" in tokenize("cartão", fold_accents=False)


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"x"}) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


def test_bm25_ranks_matching_doc_first():
    idx = BM25({"x": "gato preto", "y": "cachorro branco"})
    assert idx.search("gato")[0][0] == "x"
    assert idx.search("zebra") == []


def test_bm25_empty_corpus():
    assert BM25({}).search("a") == []


def test_accent_folding_matches_unaccented_query():
    docs = {"x": "cartão de débito", "y": "dinheiro"}
    assert BM25(docs).search("cartao")[0][0] == "x"
    assert BM25(docs, fold_accents=False).search("cartao") == []


def test_example_set_gate_and_folding_gain():
    corpus, queries = load_corpus(CORPUS), load_queries(QUERIES)
    on = evaluate(corpus, queries, k=3)
    off = evaluate(corpus, queries, k=3, fold_accents=False)
    assert on["recall@3"] >= 0.9
    assert on["recall@3"] > off["recall@3"]


def test_unknown_doc_id_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        evaluate({"a": "x"}, [{"query": "x", "relevant": ["zzz"]}])


def test_query_without_relevant_is_rejected(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps({"query": "x", "relevant": []}) + "\n")
    with pytest.raises(ValueError):
        load_queries(f)


def test_cli_gate_passes_and_fails(capsys):
    assert main(["--corpus", CORPUS, "--queries", QUERIES, "--min-recall", "0.5"]) == 0
    assert main(["--corpus", CORPUS, "--queries", QUERIES, "--no-fold", "--min-recall", "0.99"]) == 1

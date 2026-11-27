# ptbr-retrieval-eval

Demo independente, sem ligação com a Ipiranga. (Independent demo, not affiliated with Ipiranga.)

Uma ferramenta pequena para medir se a busca de um RAG em português acha o documento certo. Ela usa BM25 em Python puro, sem dependências, calcula Recall@k e MRR e pode funcionar como porta de CI: sai com código 1 se o Recall@k cair abaixo de um mínimo.

Escrevi para mostrar uma checagem que eu usaria antes de trocar o modelo de embeddings ou o chunking de um RAG: um conjunto pequeno de perguntas com resposta conhecida, rodado a cada mudança.

## Como rodar

```bash
pip install -e ".[test]"
ptbr-eval --corpus data/example_corpus.jsonl --queries data/example_queries.jsonl
ptbr-eval --corpus data/example_corpus.jsonl --queries data/example_queries.jsonl --no-fold   # sem tratar acentos
ptbr-eval ... --min-recall 0.9    # falha (exit 1) se Recall@3 < 0.9
pytest -q
```

Formato: um JSON por linha. Corpus: `{"id": "d01", "text": "..."}`. Perguntas: `{"query": "...", "relevant": ["d01"]}`.

## O que a demo mostrou

Os dados em `data/` são **exemplo inventado** por mim (10 trechos de FAQ de um posto de combustível genérico e 12 perguntas). Nesse conjunto, a busca com acentos tratados (`cartão` = `cartao`) deu Recall@3 = 1.000 e MRR = 0.958. Sem esse tratamento, deu Recall@3 = 0.833 e MRR = 0.792. Duas perguntas digitadas sem acento ("como pago com cartao", "tem cafe e lanche no posto") não acharam o documento.

## What was tested

- 11 testes offline com `pytest`: normalização de acentos, stopwords, BM25, Recall@k, MRR, rejeição de dados inválidos e o código de saída da porta de CI.
- O CI do GitHub roda esses testes e a porta de CI no conjunto de exemplo.

## What was not tested

- Dados reais. O conjunto de exemplo é pequeno e escrito por mim, então os números acima mostram que a ferramenta funciona, não o desempenho de nenhum sistema real.
- Busca vetorial, reranker ou qualquer modelo de linguagem. O BM25 é só o ponto de partida.
- Corpus grandes: o índice fica em memória e a busca percorre todos os documentos.
- Stemming (plural, conjugação). Não há.

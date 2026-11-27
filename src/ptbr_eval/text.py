"""Portuguese text normalisation: accent folding, tokenising, stopwords."""
import re
import unicodedata

STOPWORDS = frozenset(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem "
    "e ou que se ao aos como mais meu minha qual quais quando onde ser ter foi sao "
    "eh esta estao pode posso vai".split()
)


def fold(s: str) -> str:
    """Lowercase and strip accents: 'Cartão' -> 'cartao'."""
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def tokenize(s: str, *, fold_accents: bool = True, drop_stopwords: bool = True) -> list[str]:
    s = fold(s) if fold_accents else s.lower()
    toks = re.findall(r"\w+", s)
    if drop_stopwords:
        stop = STOPWORDS if fold_accents else {w for w in STOPWORDS}
        toks = [t for t in toks if t not in stop]
    return toks

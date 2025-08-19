from __future__ import annotations
from typing import List, Tuple, Set
import re

def split_style_content(prompt: str, backend: str = "spacy") -> Tuple[List[str], List[str]]:
    """
    Returns two word lists: (content_words, style_words).
    Heuristics:
     - capture phrases like 'in <STYLE> style', 'in <STYLE> aesthetic'
     - adjectives before nouns go to content unless captured as style.
     - backend="spacy": uses noun chunks; "pos": simple POS-ish regex; "naive": only the 'in ... style' rule.
    """
    p = prompt.strip()
    style_words: List[str] = []
    content_words: List[str] = []

    # 1) explicit 'in ... style' patterns
    m = re.findall(r"in ([A-Za-z0-9,\- ]+?) (?:style|aesthetic)", p, flags=re.IGNORECASE)
    if m:
        style_phrase = " ".join(m)
        style_words += [w for w in re.split(r"[ ,\-]+", style_phrase) if w]

    if backend == "spacy":
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(p)
            content_words += [t.text for t in doc if t.pos_ in {"NOUN","PROPN","VERB"}]
            # adjectives not already captured as style
            content_words += [t.text for t in doc if t.pos_ == "ADJ" and t.text.lower() not in [w.lower() for w in style_words]]
        except Exception:
            backend = "pos"

    if backend == "pos":
        toks = re.findall(r"[A-Za-z0-9']+", p)
        # crude: nouns≈words not in a small stoplist
        stop = {"in","on","the","a","an","of","and","with","to","for","style","aesthetic"}
        content_words += [w for w in toks if w.lower() not in stop and w.lower() not in [s.lower() for s in style_words]]

    if backend == "naive":
        # already filled style; everything else content
        toks = re.findall(r"[A-Za-z0-9']+", p)
        content_words += [w for w in toks if w.lower() not in [s.lower() for s in style_words]]

    # de-dup while preserving order
    def uniq(xs: List[str]) -> List[str]:
        seen: Set[str] = set(); out: List[str] = []
        for x in xs:
            xl = x.lower()
            if xl not in seen:
                seen.add(xl); out.append(x)
        return out

    return uniq(content_words), uniq(style_words)

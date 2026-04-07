from typing import Optional

import spacy

# Lazy-load the spacy model to avoid overhead if not used immediately
_nlp: Optional[spacy.Language] = None


def get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def normalize_term(term: str) -> str:
    """
    Normalizes a term for fuzzy matching by lemmatizing and lowercasing.
    Removes standard determiners (a, an, the) to make comparison robust.
    """
    doc = get_nlp()(term)

    # Filter out determiners and punctuation, keep only meaningful lemmas
    lemmas = []
    for token in doc:
        if token.pos_ not in ("DET", "PUNCT"):
            lemmas.append(token.lemma_.lower())

    # If the term was entirely determiners/punctuation (unlikely but possible),
    # fallback to the original lowercased term.
    if not lemmas:
        return term.lower().strip()

    return " ".join(lemmas).strip()


def is_singular_term(term: str) -> bool:
    """
    Determines if a term is primarily a singular entity/proper noun.
    Useful for classifying 'Socrates' as a Universal instead of Particular.
    """
    if not term:
        return False

    # Wrap the term in a pseudo-sentence to provide context for the POS tagger
    # e.g., if term is "Socrates", "Some Socrates are here." helps spacy tag it as PROPN.
    doc = get_nlp()(f"Some {term} are here.")

    # We ignore the first word ("Some") and the last words ("are here.")
    # and look at the tokens of the actual term.
    # Note: doc length is 2 (prefix) + len(term_tokens) + 3 (suffix) = len(term_tokens) + 5?
    # Actually, "Some" is 1 token, "are" is 1, "here" is 1, "." is 1. = 4 surrounding tokens.
    for token in doc[1:-3]:
        if token.pos_ in ("PROPN", "PRON"):
            return True

    return False

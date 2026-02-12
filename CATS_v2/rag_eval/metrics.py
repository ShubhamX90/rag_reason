# rag_eval/metrics.py
# -*- coding: utf-8 -*-
"""
Utility Functions for CATS v2.0 Metrics
--------------------------------------
Provides helper functions for claim extraction and basic metrics.
"""

from typing import List
from nltk import sent_tokenize
import re


def answered_flags(outputs: List[str]) -> List[bool]:
    """
    Detect if each output is a real answer (vs refusal).
    Returns list of booleans indicating if each output contains an answer.
    """
    flags = []
    for o in outputs:
        text = (o or "").strip().lower()
        
        # Check for refusal patterns
        is_refusal = (
            len(text) == 0 or
            text.startswith("i cannot") or
            text.startswith("i can't") or
            text.startswith("i am unable") or
            "cannot answer" in text or
            "can't answer" in text or
            "don't have enough information" in text or
            "insufficient information" in text
        )
        
        flags.append(not is_refusal)
    
    return flags


def extract_claims_by_sentence(answer_text: str, max_claims: int = 12) -> List[str]:
    """
    Split answer text into candidate claims (sentences).
    
    Args:
        answer_text: The model's answer text
        max_claims: Maximum number of claims to extract
    
    Returns:
        List of claim sentences (up to max_claims)
    """
    if not answer_text:
        return []
    
    try:
        # Use NLTK to tokenize sentences
        sents = sent_tokenize(answer_text or "")
        # Filter out empty sentences and strip whitespace
        sents = [s.strip() for s in sents if s.strip()]
        # Return up to max_claims
        return sents[:max_claims]
    except Exception as e:
        # Fallback: simple split on periods
        sents = [s.strip() + "." for s in answer_text.split(".") if s.strip()]
        return sents[:max_claims]


def extract_bracket_citations(answer_text: str) -> List[str]:
    """
    Extract [dX] style citations from answer text.
    
    Args:
        answer_text: Text potentially containing citations like [d1], [d2]
    
    Returns:
        List of doc IDs referenced
    """
    if not answer_text:
        return []
    
    # Find all [dX] patterns
    pattern = r'\[(d\d+)\]'
    citations = re.findall(pattern, answer_text)
    
    # Return unique citations while preserving order
    seen = set()
    unique_citations = []
    for cite in citations:
        if cite not in seen:
            unique_citations.append(cite)
            seen.add(cite)
    
    return unique_citations


def f1_gr_from_flags(pred_answered: bool, gold_answerable: bool) -> float:
    """
    Per-item F1_GR proxy (grounded refusal).
    Measures if the model correctly decided to answer or refuse.
    
    Args:
        pred_answered: Whether model provided an answer (vs refused)
        gold_answerable: Whether the question is answerable from docs
    
    Returns:
        1.0 if prediction matches gold, 0.0 otherwise
    """
    return 1.0 if int(pred_answered) == int(gold_answerable) else 0.0


def normalize_answer(text: str) -> str:
    """
    Normalize text for comparison.
    Lowercase, remove extra spaces, punctuation.
    """
    import string
    
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def remove_citations(text: str) -> str:
    """
    Remove citation markers like [1], [d1], etc. from text.
    """
    # Remove [number] or [dNumber] patterns
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[d\d+\]', '', text)
    # Clean up extra spaces
    text = ' '.join(text.split())
    return text

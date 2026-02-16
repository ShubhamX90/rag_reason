# retrieval.py
# -*- coding: utf-8 -*-
"""
Retrieval module
----------------

Adapted with gratitude from:
https://github.com/princeton-nlp/ALCE/blob/main/retrieval.py

Provides retrieval functions for evaluation:
  • GTR-based Wikipedia retrieval
  • Index building and caching

Environment variables (for customization):
  - DOCS_PICKLE: path to cached docs pickle (default: "docs.pkl")
  - DPR_WIKI_TSV: path to Wikipedia TSV (default: "psgs_w100.tsv")
  - GTR_EMB: path to cached embeddings pickle (default: "gtr_wikipedia_index.pkl")

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

import csv
import gc
import os
import pickle
from typing import Any, Dict, List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .logging_config import logger


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

def retrieve(
    queries: Union[str, List[str]], top_k: int = 5
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Retrieve top-k documents for query/queries.

    Args:
        queries: single string query or list of queries
        top_k: number of results per query

    Returns:
        If input is str → list of dicts
        If input is list[str] → list[list[dict]]
    """
    return gtr_wiki_query(queries, top_k=top_k)


def gtr_wiki_query(
    queries: Union[str, List[str]], top_k: int = 5
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Run GTR-based retrieval over Wikipedia.

    Args:
        queries: str or list[str]
        top_k: number of results per query

    Returns:
        list of dicts (if one query) or list[list[dict]] (if many)
    """
    if isinstance(queries, str):
        queries = [queries]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading GTR encoder (sentence-transformers/gtr-t5-xxl)...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device=device)

    # Encode queries
    with torch.inference_mode():
        query_embeddings_raw = encoder.encode(
            queries, batch_size=4, show_progress_bar=True, normalize_embeddings=True
        )
    query_embeddings = torch.tensor(
        query_embeddings_raw, dtype=torch.float16, device="cpu"
    )

    # Load docs
    docs = _load_or_build_docs()

    # Load or build embeddings
    embs = _load_or_build_index(encoder, docs)
    del encoder  # free GPU memory

    gtr_emb = torch.tensor(embs, dtype=torch.float16, device=device)

    # Perform retrieval
    logger.info(f"Running retrieval for {len(queries)} queries...")
    query_embeddings = query_embeddings.to(device)
    results: List[List[Dict[str, Any]]] = []

    for q_idx in range(len(queries)):
        q_emb = query_embeddings[q_idx]
        scores = torch.matmul(gtr_emb, q_emb)  # (num_docs,)
        score, idx = torch.topk(scores, top_k)

        query_results = []
        for i in range(idx.size(0)):
            try:
                title, text = docs[idx[i].item()].split("\n")
            except Exception:
                # fallback if malformed
                title, text = "Unknown", docs[idx[i].item()]
            query_results.append(
                {
                    "id": str(idx[i].item() + 1),
                    "title": title,
                    "text": text,
                    "score": float(score[i].item()),
                }
            )
        results.append(query_results)

    # Cleanup
    del gtr_emb, query_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results[0] if len(results) == 1 else results


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_or_build_docs() -> List[str]:
    """
    Load cached docs (pickle) or build from TSV if not cached.
    """
    DOCS_PICKLE = os.environ.get("DOCS_PICKLE", "docs.pkl")
    if os.path.exists(DOCS_PICKLE):
        logger.info(f"Loading docs from pickle: {DOCS_PICKLE}")
        with open(DOCS_PICKLE, "rb") as f:
            docs = pickle.load(f)
    else:
        DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV", "psgs_w100.tsv")
        logger.info(f"Processing Wikipedia TSV: {DPR_WIKI_TSV}")
        docs = []
        with open(DPR_WIKI_TSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader)  # skip header
            for row in tqdm(reader, desc="Reading Wikipedia"):
                docs.append(row[2] + "\n" + row[1])
        with open(DOCS_PICKLE, "wb") as f:
            pickle.dump(docs, f)
        logger.info(f"Saved docs to pickle: {DOCS_PICKLE}")
    return docs


def _load_or_build_index(encoder: SentenceTransformer, docs: List[str]) -> np.ndarray:
    """
    Load cached GTR embeddings or build index if not cached.
    """
    GTR_EMB = os.environ.get("GTR_EMB", "gtr_wikipedia_index.pkl")
    if os.path.exists(GTR_EMB):
        logger.info(f"Loading GTR embeddings from: {GTR_EMB}")
        with open(GTR_EMB, "rb") as f:
            embs = pickle.load(f)
    else:
        logger.info("Building GTR embeddings index...")
        embs = gtr_build_index(encoder, docs)
        with open(GTR_EMB, "wb") as f:
            pickle.dump(embs, f)
        logger.info(f"Saved embeddings to: {GTR_EMB}")
    return embs


def gtr_build_index(encoder: SentenceTransformer, docs: List[str]) -> np.ndarray:
    """
    Build embeddings for docs using the given encoder.
    """
    with torch.inference_mode():
        embs = encoder.encode(
            docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True
        )
    return embs.astype("float16")

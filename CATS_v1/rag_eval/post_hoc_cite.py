# post_hoc_cite.py
# -*- coding: utf-8 -*-
"""
Post-hoc citation assignment for model outputs
----------------------------------------------

This module re-assigns citations to model outputs after generation,
based on semantic search or NLI scoring within the retrieved docs.

Supported retrievers:
  • "gtr-*"  → SentenceTransformer (dense retrieval, e.g. gtr-t5-large)
  • "nli"    → TRUE-NLI model (via autoais loader)

Usage:
    from post_hoc_cite import post_hoc_cite
    new_data = post_hoc_cite(data, args)

Expected args namespace includes:
  - external_docs: path to JSON file with external docs (optional)
  - posthoc_retriever: retriever type string ("gtr-t5-large" or "nli")
  - posthoc_retriever_device: device string ("cpu" or "cuda")
  - ndoc: max number of docs to keep
  - data_type: string, e.g. "qampari" or other
  - overwrite: bool, whether to overwrite existing citations

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

import json
import re
from typing import Any, Dict, List

import numpy as np
import torch
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .auto_ais_loader import get_autoais_model_and_tokenizer
from .logging_config import logger
from .searcher import SearcherWithinDocs
from .utils import remove_citations


def post_hoc_cite(data: List[Dict[str, Any]], args: Any) -> List[Dict[str, Any]]:
    """
    Assign citations to model outputs after generation.

    Args:
        data: list of records, each with keys:
              - "question": str
              - "output": str (model’s generated answer)
              - "docs": list of retrieved documents
        args: argparse.Namespace or object with attributes:
              - external_docs: str path or None
              - posthoc_retriever: str ("gtr-*" or "nli")
              - posthoc_retriever_device: str
              - ndoc: int
              - data_type: str ("qampari" etc.)
              - overwrite: bool

    Returns:
        new_data: list of updated records with re-assigned citations
    """
    logger.info("Performing post hoc citation...")

    new_data: List[Dict[str, Any]] = []
    external = (
        json.load(open(args.external_docs, "r", encoding="utf-8"))
        if getattr(args, "external_docs", None) is not None
        else None
    )

    # Load retrieval model
    if "gtr" in args.posthoc_retriever:
        logger.info(f"Loading {args.posthoc_retriever}...")
        gtr_model = SentenceTransformer(
            f"sentence-transformers/{args.posthoc_retriever}",
            device=args.posthoc_retriever_device,
        )
        autoais_model, autoais_tokenizer = None, None
    elif "nli" in args.posthoc_retriever:
        logger.info("Loading TRUE-NLI model...")
        autoais_model, autoais_tokenizer = get_autoais_model_and_tokenizer(args)
        gtr_model = None
    else:
        raise ValueError(f"Unsupported retriever type: {args.posthoc_retriever}")

    for idx, item in enumerate(tqdm(data, desc="Post-hoc citation")):
        doc_list = item["docs"]

        # If external docs provided, replace
        if args.external_docs is not None:
            assert external is not None
            assert external[idx]["question"] == item["question"]
            doc_list = external[idx]["docs"][: args.ndoc]

        # Build searcher
        if "gtr" in args.posthoc_retriever:
            searcher = SearcherWithinDocs(
                doc_list,
                args.posthoc_retriever,
                model=gtr_model,
                device=args.posthoc_retriever_device,
            )
        elif "nli" in args.posthoc_retriever:
            searcher = SearcherWithinDocs(
                doc_list,
                args.posthoc_retriever,
                model=autoais_model,
                tokenizer=autoais_tokenizer,
                device=args.posthoc_retriever_device,
            )
        else:
            continue

        logger.debug(f"Original output: {item['output']}")

        # Preprocess output
        output = item["output"].replace("<|im_end|>", "").strip()
        if "qampari" in args.data_type:
            sents = [
                item["question"] + " " + x.strip()
                for x in output.rstrip(".").split(",")
            ]
        else:
            sents = sent_tokenize(output)

        # Re-assign citations
        new_output = ""
        for sent in sents:
            # Extract any existing citation indices
            original_ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]

            if len(original_ref) == 0 or args.overwrite:
                logger.debug("-----")
                logger.debug(f"Sentence before: {sent}")
                logger.debug(f"Original refs: {original_ref}")

                clean_sent = remove_citations(sent)
                best_doc_id = searcher.search(clean_sent)
                sent = f"[{best_doc_id+1}] " + clean_sent

                logger.debug(f"Assigned ref: {best_doc_id}")
                logger.debug(f"Sentence after: {sent}")

            if "qampari" in args.data_type:
                new_output += sent.replace(item["question"], "").strip() + ", "
            else:
                new_output += sent + " "

        # Update record
        item["output"] = new_output.rstrip(" ,")
        item["docs"] = doc_list
        logger.debug("Final output: " + item["output"])
        new_data.append(item)

    return new_data

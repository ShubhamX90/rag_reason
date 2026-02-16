# searcher.py
# -*- coding: utf-8 -*-
"""
Searcher Within Docs
--------------------

Provides lightweight retrieval within a given set of candidate docs,
for post-hoc citation and evaluation.

Supported retrievers:
  • TF-IDF (lexical)
  • GTR (dense SentenceTransformer encoders)
  • NLI (entailment scoring via seq2seq model)

Usage:
    searcher = SearcherWithinDocs(docs, retriever="gtr-t5-large", model=gtr_model)
    best_idx = searcher.search("What is the capital of France?")
    best_doc = docs[best_idx]

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import format_document


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def doc_to_text_tfidf(doc) -> str:
    """Concatenate title + text for TF-IDF encoding."""
    return f"{doc.get('title','')} {doc.get('text','')}".strip()


def doc_to_text_dense(doc) -> str:
    """Concatenate title + text with separator for dense embeddings."""
    return f"{doc.get('title','')}. {doc.get('text','')}".strip()


# ---------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------

class SearcherWithinDocs:
    """
    Searcher that selects the most relevant document from a provided list.

    Args:
        docs: list of document dicts (must have 'title' and 'text')
        retriever: str, one of {"tfidf", "gtr-*", "nli"}
        model: underlying model (SentenceTransformer or seq2seq)
        tokenizer: tokenizer for NLI retriever
        device: "cuda" or "cpu"
    """

    def __init__(
        self,
        docs,
        retriever: str,
        model=None,
        tokenizer=None,
        device: str = "cuda",
    ) -> None:
        self.docs = docs
        self.retriever = retriever.lower()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if self.retriever == "tfidf":
            self.tfidf = TfidfVectorizer()
            self.tfidf_docs = self.tfidf.fit_transform(
                [doc_to_text_tfidf(doc) for doc in docs]
            )
        elif "gtr" in self.retriever:
            assert model is not None, "Dense retriever requires SentenceTransformer model"
            self.embeddings = self.model.encode(
                [doc_to_text_dense(doc) for doc in docs],
                device=self.device,
                convert_to_numpy=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        elif "nli" in self.retriever:
            assert model is not None and tokenizer is not None, "NLI retriever requires model+tokenizer"
        else:
            raise NotImplementedError(f"Retriever '{retriever}' not supported")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def search(self, query: str) -> int:
        """
        Return index of top-1 doc for the given query.
        """
        if self.retriever == "tfidf":
            return self._search_tfidf(query)
        elif "gtr" in self.retriever:
            return self._search_gtr(query)
        elif "nli" in self.retriever:
            return self._search_nli(query)
        else:
            raise NotImplementedError(f"Retriever '{self.retriever}' not supported")

    # -----------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------

    def _search_tfidf(self, query: str) -> int:
        tfidf_query = self.tfidf.transform([query])
        sims = (self.tfidf_docs @ tfidf_query.T).toarray().squeeze()
        return int(np.argmax(sims))

    def _search_gtr(self, query: str) -> int:
        q_embed = self.model.encode(
            [query],
            device=self.device,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        scores = torch.matmul(self.embeddings, q_embed.t()).squeeze().cpu().numpy()
        return int(np.argmax(scores))

    def _search_nli(self, query: str) -> int:
        claim = query
        scores = np.array(
            [self.get_entailment_score(format_document(doc), claim) for doc in self.docs]
        )
        return int(np.argmax(scores))

    # -----------------------------------------------------------------
    # NLI scoring
    # -----------------------------------------------------------------

    def get_entailment_score(self, passage: str, claim: str) -> float:
        """
        Compute entailment score between passage (premise) and claim (hypothesis).
        Uses a seq2seq NLI model (e.g., t5_xxl_true_nli_mixture).
        """
        input_text = f"premise: {passage} hypothesis: {claim}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.model.device
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences
        scores = outputs.scores  # list of logits for each generated token

        log_probs = []
        for i, score in enumerate(scores):
            log_prob = torch.nn.functional.log_softmax(score, dim=-1)
            token_id = generated_ids[0, i + 1]  # skip the first input token
            log_probs.append(log_prob[0, token_id].item())

        return float(sum(log_probs))
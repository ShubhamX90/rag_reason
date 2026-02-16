# utils.py
# -*- coding: utf-8 -*-
"""
Utility functions
-----------------

Shared helpers for:
  • Text normalization
  • Citation cleanup
  • GPU memory management
  • Model loading (HF + vLLM)
  • Prompt construction for demos
  • Document formatting

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from vllm import LLM as VLLM, SamplingParams
from vllm.transformers_utils.tokenizers import MistralTokenizer

from .logging_config import logger

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast, MistralTokenizer]

# ---------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize text for comparison (lower, strip, remove articles & punctuation)."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent: str) -> str:
    """Remove bracketed citations like [1], [12] from a sentence."""
    return (
        re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent))
        .replace(" |", "")
        .replace("]", "")
    )

# ---------------------------------------------------------------------
# GPU memory utils
# ---------------------------------------------------------------------

def get_max_memory(reserve_gb: int = 3) -> Dict[int, str]:
    """
    Get the maximum memory available for each GPU, minus reserve_gb.
    Returns a dict {gpu_id: "<X>GB"} suitable for HF device_map.
    """
    if not torch.cuda.is_available():
        return {}
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory_str = f"{max(free_in_GB - reserve_gb, 1)}GB"
    n_gpus = torch.cuda.device_count()
    return {i: max_memory_str for i in range(n_gpus)}

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.float16,
    int8: bool = False,
    lora_path: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a HuggingFace causal LM + tokenizer.

    Args:
        model_name_or_path: path or HF hub ID
        dtype: torch.float16 or torch.bfloat16
        int8: whether to use int8 quantization
        lora_path: optional LoRA adapter path

    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading {model_name_or_path} with dtype={dtype}...")
    if int8:
        logger.warning("Loading with int8 quantization enabled")

    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=dtype,
        max_memory=get_max_memory(),
        load_in_8bit=int8,
    )

    if lora_path:
        from peft import PeftModel
        logger.info(f"Applying LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, model_id=lora_path)

    logger.info("Model loaded in %.2f sec." % (time.time() - start_time))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    if "opt" in model_name_or_path.lower():
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_vllm(
    model_name_or_path: str, args: Any, dtype: str = "bfloat16"
) -> Tuple[VLLM, AnyTokenizer, SamplingParams]:
    """
    Load a vLLM model for fast batched inference.

    Args:
        model_name_or_path: path or HF hub ID
        args: config/args object with {seed, max_length, temperature, top_p, max_new_tokens}
        dtype: "bfloat16" or "float16"

    Returns:
        (vllm_model, tokenizer, sampling_params)
    """
    logger.info(f"Loading vLLM model {model_name_or_path} in {dtype}...")
    start_time = time.time()
    model = VLLM(
        model_name_or_path,
        dtype=dtype,
        gpu_memory_utilization=0.9,
        seed=args.seed,
        max_model_len=args.max_length,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    logger.info("vLLM loaded in %.2f sec." % (time.time() - start_time))

    tokenizer = model.get_tokenizer()
    tokenizer.padding_side = "left"  # type: ignore[union-attr]

    return model, tokenizer, sampling_params

# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------

def make_doc_prompt(
    doc: Dict[str, Any], doc_id: int, doc_prompt: str, use_shorter: Optional[str] = None
) -> str:
    """
    Build document string for prompt.

    Placeholders:
      - {ID} → doc id (starting from 1)
      - {T}  → title
      - {P}  → text (full, summary, or extraction)

    Args:
        doc: dict with keys {title, text, summary?, extraction?}
        doc_id: integer index
        doc_prompt: template string
        use_shorter: optional "summary" or "extraction"
    """
    text = doc.get("text", "")
    if use_shorter is not None and use_shorter in doc:
        text = doc[use_shorter]

    return (
        doc_prompt.replace("{T}", doc.get("title", ""))
        .replace("{P}", text)
        .replace("{ID}", str(doc_id + 1))
    )


def get_shorter_text(
    docs: List[Dict[str, Any]], ndoc: int, key: str
) -> List[Dict[str, Any]]:
    """
    Select up to ndoc documents that contain a shorter version (summary/extraction).
    Falls back to full text if not available.
    """
    doc_list: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        if key not in doc:
            if len(doc_list) == 0:
                # ensure at least one doc is provided
                doc[key] = doc.get("text", "")
                doc_list.append(doc)
            logger.warning(
                f"No {key} found in document {i}. "
                f"Only {len(doc_list)} documents will be used."
            )
            break
        if "irrelevant" in doc[key].lower():
            continue
        doc_list.append(doc)
        if len(doc_list) >= ndoc:
            break
    return doc_list


def make_demo(
    item: Dict[str, Any],
    prompt: str,
    ndoc: int,
    doc_prompt: str,
    instruction: str = "",
    use_shorter: Optional[str] = None,
    test: bool = False,
) -> str:
    """
    Construct a demo example for prompting.

    Placeholders:
      - {INST}: instruction
      - {Q}:    question
      - {D}:    documents
      - {A}:    answer(s)

    Args:
        item: dataset item with keys {question, docs, answer}
        prompt: template string
        ndoc: number of docs to include
        doc_prompt: template for each doc
        instruction: optional instruction string
        use_shorter: "summary" or "extraction"
        test: if True, omit the answer
    """
    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item.get("question", ""))

    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "")
        else:
            doc_list = (
                get_shorter_text(item.get("docs", []), ndoc, use_shorter)
                if use_shorter is not None
                else item.get("docs", [])[:ndoc]
            )
            text = "".join(
                make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter)
                for doc_id, doc in enumerate(doc_list)
            )
            prompt = prompt.replace("{D}", text)

    if not test:
        ans = item.get("answer", "")
        if isinstance(ans, list):
            ans = "\n" + "\n".join(ans)
        prompt = prompt.replace("{A}", "").rstrip() + str(ans)
    else:
        prompt = prompt.replace("{A}", "").rstrip()

    return prompt

# ---------------------------------------------------------------------
# Document formatting
# ---------------------------------------------------------------------

def format_document(doc: Dict[str, Any]) -> str:
    """
    Format document for NLI scoring / AutoAIS style.

    Args:
        doc: dict with keys {title, text, sent?}
    """
    if "sent" in doc:
        return f"Title: {doc.get('title','')}\n{doc.get('sent','')}"
    return f"Title: {doc.get('title','')}\n{doc.get('text','')}"

#!/usr/bin/env python3
"""Fast compatibility checks for local HF model dirs before long Slurm runs."""

import argparse
import importlib.util
import importlib
import os
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


def content_text(msg):
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def as_text_parts(msgs):
    converted = []
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({**msg, "content": content})
    return converted


def manual_inst_chat_template(tok, msgs, *, add_generation_prompt=True):
    system_txt = ""
    user_parts = []
    for msg in msgs:
        role = msg.get("role")
        text = content_text(msg).strip()
        if not text:
            continue
        if role == "system":
            system_txt = text
        elif role == "user":
            user_parts.append(text)
        elif role == "assistant":
            user_parts.append(text)
    prompt_body = "\n\n".join([p for p in [system_txt, "\n\n".join(user_parts)] if p]).strip()
    rendered = f"{tok.bos_token or ''}[INST] {prompt_body} [/INST]"
    if add_generation_prompt:
        rendered += " "
    return rendered


def render_chat(tok):
    msgs = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User prompt."},
    ]
    attempts = [
        ("native", msgs),
        ("text-parts", as_text_parts(msgs)),
    ]
    folded = [
        {
            "role": "user",
            "content": (content_text(msgs[0]).strip() + "\n\n" + content_text(msgs[1])).strip(),
        }
    ]
    attempts.extend([
        ("folded-system-user", folded),
        ("folded-system-user-text-parts", as_text_parts(folded)),
    ])

    errors = []
    for mode, candidate in attempts:
        try:
            rendered = tok.apply_chat_template(candidate, tokenize=False, add_generation_prompt=True)
            return mode, rendered
        except Exception as exc:
            errors.append(f"{mode}: {exc}")
    return "manual-inst-fallback", manual_inst_chat_template(tok, msgs, add_generation_prompt=True)


def _load_mistral_common_tokenizer(path):
    candidates = []
    for module_name in (
        "transformers",
        "transformers.models.mistral3",
        "transformers.models.mistral3.tokenization_mistral_common",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr_name in ("MistralCommonTokenizer", "MistralCommonTokenizerFast", "MistralCommonBackend"):
            candidate = getattr(module, attr_name, None)
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)

    errors = []
    for candidate in candidates:
        for kwargs in (
            {"local_files_only": True},
            {},
        ):
            try:
                return candidate.from_pretrained(path, **kwargs)
            except Exception as exc:
                errors.append(f"{candidate.__name__}({kwargs}): {exc}")

    if candidates:
        raise RuntimeError("failed to load mistral common tokenizer: " + " | ".join(errors))
    raise RuntimeError("mistral3 detected but no Mistral common tokenizer class is importable in this transformers build")


def load_tokenizer_compat(path):
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    if getattr(config, "model_type", None) == "mistral3":
        return _load_mistral_common_tokenizer(path)

    kwargs = dict(
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    try:
        return AutoTokenizer.from_pretrained(path, **kwargs)
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(path, **kwargs)


def model_class_support(config):
    support = []
    errors = []
    if AutoModelForCausalLM is not None:
        try:
            cls = AutoModelForCausalLM._model_mapping[type(config)]
            support.append(("AutoModelForCausalLM", cls.__name__))
        except Exception as exc:
            errors.append(f"AutoModelForCausalLM: {exc}")
    if AutoModelForImageTextToText is not None:
        try:
            cls = AutoModelForImageTextToText._model_mapping[type(config)]
            support.append(("AutoModelForImageTextToText", cls.__name__))
        except Exception as exc:
            errors.append(f"AutoModelForImageTextToText: {exc}")
    return support, errors


def check_model(alias, path, *, require_fast_offsets=True):
    path = Path(path)
    print(f"===== {alias} =====")
    print(f"path={path}")
    if not path.is_dir():
        raise FileNotFoundError(f"model directory not found: {path}")

    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    print(f"model_type={getattr(config, 'model_type', None)}")
    print(f"architectures={getattr(config, 'architectures', None)}")
    print(f"max_position_embeddings={getattr(config, 'max_position_embeddings', None)}")

    if "mistral-small-3.2" in path.name.lower() or getattr(config, "model_type", "") == "mistral3":
        has_mistral_common = importlib.util.find_spec("mistral_common") is not None
        print(f"mistral_common_available={has_mistral_common}")

    tok = load_tokenizer_compat(path)
    print(f"tokenizer_class={tok.__class__.__name__}")
    print(f"is_fast={getattr(tok, 'is_fast', None)} pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id}")
    allow_missing_offsets = getattr(config, "model_type", None) == "mistral3"
    if require_fast_offsets and not getattr(tok, "is_fast", False) and not allow_missing_offsets:
        raise RuntimeError("trainer requires a fast tokenizer with offset_mapping support")
    if require_fast_offsets and not getattr(tok, "is_fast", False) and allow_missing_offsets:
        print("offset_mapping_requirement_relaxed_for_mistral3=True")

    mode, rendered = render_chat(tok)
    print(f"chat_template_mode={mode} rendered_chars={len(rendered)} preview={rendered[:160].replace(chr(10), ' ')}")
    try:
        encoded = tok(
            rendered + "Assistant target.",
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        offset_mapping_ok = bool(encoded.get("offset_mapping"))
        token_count = len(encoded["input_ids"])
    except Exception as exc:
        encoded = tok(
            rendered + "Assistant target.",
            add_special_tokens=False,
            verbose=False,
        )
        offset_mapping_ok = False
        token_count = len(encoded["input_ids"])
        print(f"offset_mapping_error={type(exc).__name__}: {exc}")
    print(f"offset_mapping_ok={offset_mapping_ok} token_count={token_count}")

    support, errors = model_class_support(config)
    print(f"model_loader_support={support}")
    if errors:
        print(f"model_loader_errors={errors}")
    if not support:
        raise RuntimeError("no supported AutoModel loader found for current trainer/generator path")
    print(f"preflight_ok={alias}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="alias=/path/to/model")
    ap.add_argument("--allow-slow-tokenizer", action="store_true")
    args = ap.parse_args()

    failures = []
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"model spec must be alias=/path, got: {spec}")
        alias, path = spec.split("=", 1)
        try:
            check_model(alias, path, require_fast_offsets=not args.allow_slow_tokenizer)
        except Exception as exc:
            failures.append((alias, exc))
            print(f"preflight_failed={alias}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print("===== PREFLIGHT FAILURES =====", file=sys.stderr)
        for alias, exc in failures:
            print(f"{alias}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

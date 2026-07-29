#!/usr/bin/env python3
"""
Retrieve documents for benchmark candidate queries using the CONFLICTS recipe.

Default method:
  1. Search top-10 results per query. Google Custom Search matches the paper;
     Tavily is the preferred no-card/free-credit alternative; DuckDuckGo HTML
     is available as a no-key fallback.
  2. Fetch full HTML for each result, preferring cloudscraper when installed.
  3. Extract readable page text with jusText when installed.
  4. Select the most relevant 512-token window with 256-token stride using TAS-B.

The output schema is the raw benchmark schema expected by
scripts/prepare_benchmark_stagewise_input.py. The conflict_type fields are
placeholders because gold labels are assigned downstream by committee/human
annotation after retrieval, matching the CONFLICTS paper's annotation order.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "data" / "benchmark_build" / "retrieval_cache"

GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
DUCKDUCKGO_ENDPOINT = "https://html.duckduckgo.com/html/"
PLACEHOLDER_CONFLICT_TYPE = "No conflict"
PLACEHOLDER_REASON = "Placeholder label; run benchmark-mode Stage-2 to annotate conflict type from retrieved evidence."


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        if row.get("id"):
            done.add(str(row["id"]))
    return done


def safe_text(value: Any) -> str:
    return html.unescape(str(value or "")).strip()


def cached_results(payload: Any) -> List[Dict[str, Any]]:
    """Return result dicts from either old list caches or new metadata caches."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        results = payload.get("results", [])
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    return []


def extract_google_date(item: Dict[str, Any]) -> str:
    pagemap = item.get("pagemap") or {}
    metatags = pagemap.get("metatags") or []
    date_keys = [
        "article:published_time",
        "article:modified_time",
        "og:updated_time",
        "datepublished",
        "datemodified",
        "date",
        "dc.date",
        "pubdate",
    ]
    for meta in metatags:
        lower = {str(k).lower(): v for k, v in meta.items()}
        for key in date_keys:
            if lower.get(key):
                return safe_text(lower[key])
    snippet = safe_text(item.get("snippet", ""))
    match = re.match(r"([A-Z][a-z]{2} \d{1,2}, \d{4})\s*[.-]\s*", snippet)
    return match.group(1) if match else ""


def extract_result_date(item: Dict[str, Any]) -> str:
    if item.get("source_provider") == "google":
        return extract_google_date(item)
    if item.get("published_date"):
        return safe_text(item.get("published_date"))
    snippet = safe_text(item.get("snippet", ""))
    match = re.match(r"([A-Z][a-z]{2} \d{1,2}, \d{4})\s*[.-]\s*", snippet)
    return match.group(1) if match else ""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def fallback_extract_text(html_text: str) -> str:
    parser = _TextExtractor()
    parser.feed(html_text)
    return parser.text()


def strip_xml_incompatible_chars(text: str) -> str:
    # lxml rejects NULL bytes and most C0 control characters during jusText cleanup.
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text or "")


def extract_page_text(html_text: str) -> str:
    html_text = strip_xml_incompatible_chars(html_text)
    try:
        import justext
    except ImportError:
        return fallback_extract_text(html_text)

    try:
        paragraphs = justext.justext(html_text, justext.get_stoplist("English"))
        text_parts = [p.text for p in paragraphs if not p.is_boilerplate and p.text]
        text = "\n".join(text_parts).strip()
        return text or fallback_extract_text(html_text)
    except Exception:
        return fallback_extract_text(html_text)


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._current: Optional[Dict[str, str]] = None
        self._capture: Optional[str] = None
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        class_name = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in class_name:
            href = attrs_dict.get("href", "")
            self._current = {
                "link": self._clean_url(href),
                "title": "",
                "snippet": "",
                "source_provider": "duckduckgo_html",
            }
            self._capture = "title"
            self._parts = []
        elif self._current is not None and "result__snippet" in class_name:
            self._capture = "snippet"
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag == "a" and self._capture == "title":
            self._current["title"] = safe_text(" ".join(self._parts))
            self._capture = None
            self._parts = []
        elif self._capture == "snippet" and tag in {"a", "div"}:
            self._current["snippet"] = safe_text(" ".join(self._parts))
            if self._current.get("link") and self._current.get("title"):
                self.results.append(self._current)
            self._current = None
            self._capture = None
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self._parts.append(text)

    @staticmethod
    def _clean_url(href: str) -> str:
        href = safe_text(href)
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return unquote(target)
        return href


class LexicalWindowSelector:
    def __init__(self, chunk_size: int, stride: int) -> None:
        self.chunk_size = chunk_size
        self.stride = stride

    def select(self, query: str, text: str) -> tuple[str, Optional[float]]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return text.strip(), None
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        best_score = -1.0
        best_window = words[: self.chunk_size]
        for start in range(0, max(1, len(words) - self.chunk_size + 1), self.stride):
            window = words[start: start + self.chunk_size]
            terms = set(re.findall(r"[a-z0-9]+", " ".join(window).lower()))
            score = len(query_terms & terms) / max(1, len(query_terms))
            if score > best_score:
                best_score = score
                best_window = window
        return " ".join(best_window), best_score


class TASBWindowSelector:
    def __init__(self, chunk_size: int, stride: int, device: str) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "TAS-B windowing requires torch and transformers. "
                "Install retrieval dependencies with: python3 -m pip install -r requirements_retrieval.txt "
                "or rerun with --window-selector lexical for a non-paper fallback."
            ) from exc

        self.torch = torch
        self.chunk_size = chunk_size
        self.stride = stride
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco", use_fast=True)
        self.tokenizer.model_max_length = 10**9
        self.model = AutoModel.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco").to(device)
        self.model_max_length = int(getattr(self.model.config, "max_position_embeddings", 512) or 512)
        self.model.eval()

    def select(self, query: str, text: str) -> tuple[str, Optional[float]]:
        torch = self.torch
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = tokenized.get("offset_mapping") or []
        offsets = [offset for offset in offsets if offset and offset[1] > offset[0]]
        effective_chunk = max(1, self.chunk_size - 2)

        if not offsets:
            return text.strip(), None
        if len(offsets) <= effective_chunk:
            windows_text = [text.strip()]
        else:
            windows_text = []
            for start in range(0, len(offsets), self.stride):
                end = min(start + effective_chunk, len(offsets))
                char_start = offsets[start][0]
                char_end = offsets[end - 1][1]
                window_text = text[char_start:char_end].strip()
                if window_text:
                    windows_text.append(window_text)
                if end == len(offsets):
                    break

        with torch.no_grad():
            q_tok = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_max_length,
            ).to(self.device)
            q_vec = self.model(**q_tok)[0][:, 0, :].squeeze(0).detach().cpu()
            encs = []
            for start in range(0, len(windows_text), 16):
                batch = windows_text[start: start + 16]
                w_tok = self.tokenizer(
                    batch,
                    add_special_tokens=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_max_length,
                ).to(self.device)
                encs.append(self.model(**w_tok)[0][:, 0, :].detach().cpu())
            scores = torch.matmul(q_vec, torch.cat(encs, dim=0).T)
            best_idx = int(torch.argmax(scores).item())
        return windows_text[best_idx], float(scores[best_idx].item())


@dataclass
class SearchConfig:
    api_key: str
    cse_id: str
    top_k: int
    delay_seconds: float
    cache_dir: Path


class GoogleCustomSearcher:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.search_cache = config.cache_dir / "google_search"
        self.search_cache.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)

    def search(self, query: str) -> List[Dict[str, Any]]:
        cache_path = self.search_cache / f"{short_hash(query)}.json"
        if cache_path.exists():
            return cached_results(json.loads(cache_path.read_text(encoding="utf-8")))

        params = {
            "key": self.config.api_key,
            "cx": self.config.cse_id,
            "q": query,
            "num": min(10, self.config.top_k),
            "start": 1,
        }
        response = self.client.get(GOOGLE_ENDPOINT, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Google Search failed for {query!r}: {response.status_code} {response.text[:500]}")
        items = response.json().get("items", [])[: self.config.top_k]
        cache_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(self.config.delay_seconds)
        return items


class TavilySearcher:
    def __init__(
        self,
        api_keys: List[str],
        top_k: int,
        delay_seconds: float,
        cache_dir: Path,
        include_raw_content: bool,
        exclude_domains: List[str],
        max_attempts: int = 0,
    ) -> None:
        if not api_keys:
            raise ValueError("at least one Tavily API key is required")
        self.api_keys = api_keys
        self.key_index = 0
        self.top_k = top_k
        self.delay_seconds = delay_seconds
        self.include_raw_content = include_raw_content
        self.exclude_domains = exclude_domains
        self.max_attempts = max_attempts or max(3, len(api_keys) * 2)
        self.search_cache = cache_dir / "tavily_search"
        self.search_cache.mkdir(parents=True, exist_ok=True)
        self.last_cache_hit: Optional[bool] = None
        self.last_usage: Dict[str, Any] = {}
        self.last_request_id = ""
        self.last_attempt_count = 0
        self.last_statuses: List[Any] = []
        self.client = httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={
                "Content-Type": "application/json",
            },
        )

    def next_key(self) -> str:
        key = self.api_keys[self.key_index % len(self.api_keys)]
        self.key_index += 1
        return key

    def search(self, query: str) -> List[Dict[str, Any]]:
        cache_path = self.search_cache / f"{short_hash(query)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.last_cache_hit = True
            self.last_usage = cached.get("usage", {}) if isinstance(cached, dict) else {}
            self.last_request_id = cached.get("request_id", "") if isinstance(cached, dict) else ""
            self.last_attempt_count = 0
            self.last_statuses = []
            return cached_results(cached)

        payload = {
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": self.top_k,
            "include_answer": False,
            "include_raw_content": "text" if self.include_raw_content else False,
            "include_images": False,
            "include_usage": True,
        }
        if self.exclude_domains:
            payload["exclude_domains"] = self.exclude_domains
        response = None
        errors: List[str] = []
        self.last_cache_hit = False
        self.last_usage = {}
        self.last_request_id = ""
        self.last_attempt_count = 0
        self.last_statuses = []
        for _ in range(self.max_attempts):
            api_key = self.next_key()
            self.last_attempt_count += 1
            try:
                response = self.client.post(
                    TAVILY_ENDPOINT,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            except httpx.RequestError as exc:
                error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
                errors.append(error)
                self.last_statuses.append(error)
                time.sleep(min(5.0, 0.5 * self.last_attempt_count))
                continue
            self.last_statuses.append(response.status_code)
            if response.status_code == 200:
                break
            errors.append(f"{response.status_code} {response.text[:200]}")
            if response.status_code not in {401, 402, 403, 429}:
                break
        if response is None or response.status_code != 200:
            raise RuntimeError(f"Tavily search failed for {query!r}: {' | '.join(errors)[:500]}")

        data = response.json()
        self.last_usage = data.get("usage", {})
        self.last_request_id = data.get("request_id", "")
        items: List[Dict[str, Any]] = []
        for result in data.get("results", [])[: self.top_k]:
            items.append({
                "link": result.get("url", ""),
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "snippet": result.get("content", ""),
                "raw_content": result.get("raw_content") or "",
                "score": result.get("score"),
                "published_date": result.get("published_date", ""),
                "source_provider": "tavily",
            })

        cache_payload = {
            "query": data.get("query", query),
            "results": items,
            "usage": data.get("usage", {}),
            "response_time": data.get("response_time"),
            "request_id": data.get("request_id"),
        }
        cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(self.delay_seconds)
        return items


class DuckDuckGoHTMLSearcher:
    def __init__(self, top_k: int, delay_seconds: float, cache_dir: Path) -> None:
        self.top_k = top_k
        self.delay_seconds = delay_seconds
        self.search_cache = cache_dir / "duckduckgo_search"
        self.search_cache.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 CONFLICTS benchmark builder",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

    def search(self, query: str) -> List[Dict[str, Any]]:
        cache_path = self.search_cache / f"{short_hash(query)}.json"
        if cache_path.exists():
            return cached_results(json.loads(cache_path.read_text(encoding="utf-8")))

        response = self.client.post(DUCKDUCKGO_ENDPOINT, data={"q": query})
        if response.status_code != 200:
            raise RuntimeError(f"DuckDuckGo HTML search failed for {query!r}: {response.status_code} {response.text[:300]}")

        parser = DuckDuckGoHTMLParser()
        parser.feed(response.text)
        items = parser.results[: self.top_k]
        cache_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(self.delay_seconds)
        return items


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def looks_blocked_or_low_signal(doc: Dict[str, Any]) -> bool:
    text = " ".join([
        safe_text(doc.get("snippet", "")),
        safe_text(doc.get("_search_snippet", "")),
    ]).lower()
    patterns = [
        "something went wrong",
        "this browser isn't supported",
        "this browser isn",
        "log in",
        "sign up",
        "enable javascript",
        "access denied",
        "just a moment",
        "please enable cookies",
        "captcha",
        "are you a robot",
    ]
    return any(pattern in text for pattern in patterns)


def looks_like_boilerplate_window(text: str) -> bool:
    text = safe_text(text).strip()
    lowered = text.lower()
    if not text:
        return False
    patterns = [
        "reddit - please wait for verification",
        "about press copyright contact us creators advertise developers terms privacy policy",
        "agree & join linkedin",
        "by clicking continue to join or sign in, you agree to linkedin",
        "cookies on gov.uk",
        "we use some essential cookies",
        "sign in what's on your mind? google offered in",
    ]
    if any(pattern in lowered for pattern in patterns):
        return True
    if re.search(r"[\ufffd�▒]{3,}", text):
        return True
    if re.search(r"^.{1,120}\s+on JSTOR$", text, flags=re.IGNORECASE):
        return True
    return False


def domain_matches(url: str, excluded_domains: List[str]) -> bool:
    if not excluded_domains:
        return False
    host = (urlparse(safe_text(url)).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return any(host == domain or host.endswith(f".{domain}") for domain in excluded_domains)


def filter_docs(
    docs: List[Dict[str, Any]],
    keep_top_k: int,
    min_window_words: int,
    drop_blocked: bool,
    exclude_domains: List[str],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for doc in docs:
        url = doc.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        if min_window_words and token_count(doc.get("snippet", "")) < min_window_words:
            doc["_quality_filter_reject_reason"] = "too_few_window_words"
            continue
        if drop_blocked and looks_blocked_or_low_signal(doc):
            doc["_quality_filter_reject_reason"] = "blocked_or_low_signal"
            continue
        if domain_matches(url, exclude_domains):
            doc["_quality_filter_reject_reason"] = "excluded_domain"
            continue
        kept.append(doc)
        if len(kept) >= keep_top_k:
            break
    return kept


class PageFetcher:
    def __init__(self, cache_dir: Path, timeout: float, delay_seconds: float) -> None:
        self.cache_dir = cache_dir / "pages"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self._cloudscraper = None
        try:
            import cloudscraper
            self._cloudscraper = cloudscraper.create_scraper()
        except ImportError:
            self._cloudscraper = None
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 CONFLICTS benchmark builder"},
        )

    def fetch(self, url: str) -> tuple[str, str]:
        cache_path = self.cache_dir / f"{short_hash(url)}.html"
        meta_path = self.cache_dir / f"{short_hash(url)}.json"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="ignore"), "cache"

        try:
            if self._cloudscraper is not None:
                response = self._cloudscraper.get(url, timeout=self.timeout)
                status = response.status_code
                text = response.text
            else:
                response = self.client.get(url)
                status = response.status_code
                text = response.text
            meta_path.write_text(json.dumps({"url": url, "status": status}, indent=2), encoding="utf-8")
            if status >= 400:
                return "", f"http_{status}"
            cache_path.write_text(text, encoding="utf-8", errors="ignore")
            time.sleep(self.delay_seconds)
            return text, "fetched"
        except Exception as exc:
            meta_path.write_text(json.dumps({"url": url, "error": str(exc)}, indent=2), encoding="utf-8")
            return "", f"error:{str(exc)[:100]}"


def build_doc(
    record_id: str,
    rank: int,
    item: Dict[str, Any],
    query: str,
    fetcher: PageFetcher,
    selector: Any,
    skip_fulltext: bool,
) -> Dict[str, Any]:
    url = safe_text(item.get("link") or item.get("url", ""))
    title = safe_text(item.get("title", ""))
    search_snippet = safe_text(item.get("snippet") or item.get("content", ""))
    date = extract_result_date(item)
    full_text = safe_text(item.get("raw_content", ""))
    fetch_status = "provided_raw_content" if full_text else "skipped"
    window_score: Optional[float] = None

    if url and not skip_fulltext and not full_text:
        html_text, fetch_status = fetcher.fetch(url)
        if html_text:
            full_text = extract_page_text(html_text)

    basis_text = full_text or search_snippet
    short_text, window_score = selector.select(query, basis_text) if basis_text else ("", None)
    snippet = short_text or search_snippet
    snippet_fallback_reason = ""
    if search_snippet and looks_like_boilerplate_window(snippet):
        snippet = search_snippet
        short_text = search_snippet
        snippet_fallback_reason = "search_snippet_used_for_boilerplate_window"

    return {
        "doc_id": f"{record_id}_doc_{rank}",
        "title": title,
        # Pipeline-facing field: the selected query-relevant window.
        "snippet": snippet,
        # CONFLICTS-style provenance fields: full extracted page text and the selected window.
        "response_str": full_text,
        "short_text": snippet,
        "url": url,
        "date": date,
        "source_url": url,
        "timestamp": date,
        "_rank": rank,
        "_search_snippet": search_snippet,
        "_search_provider": item.get("source_provider", "google"),
        "_search_score": item.get("score"),
        "_fetch_status": fetch_status,
        "_full_text_chars": len(full_text),
        "_window_score": window_score,
        "_snippet_fallback_reason": snippet_fallback_reason,
    }


def candidate_to_raw_record(candidate: Dict[str, Any], docs: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    return {
        "id": candidate["candidate_id"],
        "query": candidate["query"],
        "retrieved_docs": docs,
        "conflict_category_id": 0,
        "conflict_type": PLACEHOLDER_CONFLICT_TYPE,
        "conflict_reason": PLACEHOLDER_REASON,
        "gold_answer": candidate.get("source_answer", ""),
        "_candidate_source": {
            "source_family": candidate.get("source_family", ""),
            "source_dataset": candidate.get("source_dataset", ""),
            "source_record_id": candidate.get("source_record_id", ""),
            "source_split": candidate.get("source_split", ""),
            "source_metadata": candidate.get("source_metadata", {}),
        },
        "_retrieval_metadata": {
            "method": method,
            "top_k": len(docs),
            "placeholder_label": True,
        },
    }


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve CONFLICTS-style search documents for candidate benchmark queries")
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "benchmark_build" / "candidates" / "query_pool_2000.jsonl"))
    ap.add_argument("--output", default=str(PROJECT_ROOT / "data" / "benchmark_build" / "retrieved" / "benchmark2000_retrieved.jsonl"))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--search-provider", choices=["tavily", "duckduckgo", "google"], default="tavily")
    ap.add_argument("--tavily-api-key", default=os.environ.get("TAVILY_API_KEY", ""))
    ap.add_argument("--tavily-api-keys", default=os.environ.get("TAVILY_API_KEYS", ""),
                    help="Comma-separated Tavily keys; used round-robin and falls through on quota/rate-limit errors")
    ap.add_argument("--tavily-max-attempts", type=int, default=0,
                    help="Max Tavily attempts per uncached query; defaults to max(3, 2 * number of keys)")
    ap.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY", ""))
    ap.add_argument("--google-cse-id", default=os.environ.get("GOOGLE_CSE_ID", ""))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--search-max-results", type=int, default=None,
                    help="Number of search results to request before filtering; defaults to --top-k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--delay-seconds", type=float, default=0.7)
    ap.add_argument("--fetch-timeout", type=float, default=20.0)
    ap.add_argument("--skip-fulltext", action="store_true", help="Use Google snippets only; not the CONFLICTS recipe")
    ap.add_argument("--window-selector", choices=["tasb", "lexical"], default="tasb")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--exclude-domains", default="",
                    help="Comma-separated domains to exclude at search time when provider supports it")
    ap.add_argument("--drop-blocked", action="store_true", default=False,
                    help="Drop docs whose selected text looks like login/captcha/browser-block pages")
    ap.add_argument("--min-window-words", type=int, default=0,
                    help="Drop docs with selected windows shorter than this many word tokens")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    search_max_results = args.search_max_results or args.top_k
    exclude_domains = parse_csv_arg(args.exclude_domains)
    if args.search_provider == "google":
        if not args.google_api_key or not args.google_cse_id:
            raise SystemExit("Set GOOGLE_API_KEY and GOOGLE_CSE_ID, or pass --google-api-key and --google-cse-id.")
        searcher = GoogleCustomSearcher(SearchConfig(
            api_key=args.google_api_key,
            cse_id=args.google_cse_id,
            top_k=search_max_results,
            delay_seconds=args.delay_seconds,
            cache_dir=cache_dir,
        ))
    elif args.search_provider == "tavily":
        tavily_keys = parse_csv_arg(args.tavily_api_keys) or parse_csv_arg(args.tavily_api_key)
        if not tavily_keys:
            raise SystemExit("Set TAVILY_API_KEY/TAVILY_API_KEYS, pass --tavily-api-key/--tavily-api-keys, or use --search-provider duckduckgo for no-key retrieval.")
        searcher = TavilySearcher(
            api_keys=tavily_keys,
            top_k=search_max_results,
            delay_seconds=args.delay_seconds,
            cache_dir=cache_dir,
            include_raw_content=not args.skip_fulltext,
            exclude_domains=exclude_domains,
            max_attempts=args.tavily_max_attempts,
        )
    else:
        searcher = DuckDuckGoHTMLSearcher(
            top_k=search_max_results,
            delay_seconds=args.delay_seconds,
            cache_dir=cache_dir,
        )
    fetcher = PageFetcher(cache_dir=cache_dir, timeout=args.fetch_timeout, delay_seconds=args.delay_seconds)
    selector = (
        TASBWindowSelector(args.chunk_size, args.stride, args.device)
        if args.window_selector == "tasb"
        else LexicalWindowSelector(args.chunk_size, args.stride)
    )
    method = f"{args.search_provider}_top{args.top_k}_from{search_max_results}_fullhtml_cloudscraper_justext_{args.window_selector}{args.chunk_size}_stride{args.stride}"
    if args.skip_fulltext:
        method = f"{args.search_provider}_top{args.top_k}_from{search_max_results}_snippet_only_{args.window_selector}{args.chunk_size}_stride{args.stride}"

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_processed_ids(output_path) if args.resume else set()
    candidates = list(read_jsonl(input_path))
    if args.limit is not None:
        candidates = candidates[: args.limit]

    mode = "a" if args.resume else "w"
    processed = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for idx, cand in enumerate(candidates, start=1):
            record_id = cand["candidate_id"]
            if record_id in done:
                continue
            print(f"[{idx}/{len(candidates)}] {record_id}: {cand['query']}", file=sys.stderr)
            items = searcher.search(cand["query"])
            raw_docs = [
                build_doc(record_id, rank, item, cand["query"], fetcher, selector, args.skip_fulltext)
                for rank, item in enumerate(items, start=1)
            ]
            raw_docs = [doc for doc in raw_docs if doc["snippet"] or doc["url"]]
            docs = filter_docs(
                raw_docs,
                keep_top_k=args.top_k,
                min_window_words=args.min_window_words,
                drop_blocked=args.drop_blocked,
                exclude_domains=exclude_domains,
            )
            record = candidate_to_raw_record(cand, docs, method)
            record["_retrieval_metadata"]["requested_results"] = search_max_results
            record["_retrieval_metadata"]["raw_docs_before_filter"] = len(raw_docs)
            record["_retrieval_metadata"]["exclude_domains"] = exclude_domains
            record["_retrieval_metadata"]["drop_blocked"] = args.drop_blocked
            record["_retrieval_metadata"]["min_window_words"] = args.min_window_words
            if args.search_provider == "tavily":
                record["_retrieval_metadata"]["search_cache_hit"] = getattr(searcher, "last_cache_hit", None)
                record["_retrieval_metadata"]["tavily_usage"] = getattr(searcher, "last_usage", {})
                record["_retrieval_metadata"]["tavily_request_id"] = getattr(searcher, "last_request_id", "")
                record["_retrieval_metadata"]["tavily_attempt_count"] = getattr(searcher, "last_attempt_count", 0)
                record["_retrieval_metadata"]["tavily_statuses"] = getattr(searcher, "last_statuses", [])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            processed += 1

    print(f"wrote {processed} new records to {output_path}")


if __name__ == "__main__":
    main()

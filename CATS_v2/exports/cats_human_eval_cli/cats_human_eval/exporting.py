from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .storage import list_active_judgments
from .study import load_manifest, sample_index


def export_active_judgments(study_dir: Path) -> Dict[str, str]:
    manifest = load_manifest(study_dir)
    samples = sample_index(study_dir)
    judgments = list_active_judgments(study_dir)
    out_dir = study_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "active_judgments.jsonl"
    enriched_path = out_dir / "active_judgments_enriched.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_handle, enriched_path.open("w", encoding="utf-8") as enriched_handle:
        for judgment in judgments:
            raw_handle.write(json.dumps(judgment, ensure_ascii=False) + "\n")
            sample = samples.get(judgment["sample_id"], {})
            enriched = {
                "study_name": manifest["study_name"],
                "reviewer_id": judgment["reviewer_id"],
                "sample_id": judgment["sample_id"],
                "status": judgment["status"],
                "revision": judgment["revision"],
                "query": sample.get("query"),
                "conflict_category_id": sample.get("conflict_category_id"),
                "conflict_type": sample.get("conflict_type"),
                "gold_answerable": sample.get("gold_answerable"),
                "correct_refusal": sample.get("correct_refusal"),
                "claims_with_citations": sample.get("claims_with_citations"),
                "judgment": judgment["payload"],
            }
            enriched_handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    return {"raw": str(raw_path), "enriched": str(enriched_path)}

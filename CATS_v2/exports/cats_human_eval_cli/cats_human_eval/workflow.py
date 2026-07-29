from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from rich.prompt import Confirm, IntPrompt, Prompt

from .logic import compute_fg_claim_support
from .render import (
    app_console,
    render_behavior_guide,
    render_claim,
    render_dashboard,
    render_docs,
    render_fg_guide,
    render_help,
    render_review_summary,
    render_sample_overview,
    render_str_guide,
)
from .storage import append_event, get_assignment_progress, get_latest_judgment, save_judgment
from .study import load_manifest, sample_index

_CONFIDENCE_LEVELS: Dict[int, Tuple[str, float]] = {
    1: ("low", 0.25),
    2: ("medium-low", 0.50),
    3: ("medium-high", 0.75),
    4: ("high", 1.00),
}


def _blank_draft(sample: Dict[str, Any], reviewer: str) -> Dict[str, Any]:
    return {
        "reviewer_id": reviewer,
        "sample_id": sample["sample_id"],
        "meta": {"notes": ""},
        "behavior": None,
        "fg": None,
        "str": None,
    }


def _seed_draft(sample: Dict[str, Any], reviewer: str, latest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if latest is None:
        return _blank_draft(sample, reviewer)
    return deepcopy(latest["payload"])


def _prompt_rationale(label: str, default: str = "") -> str:
    while True:
        value = Prompt.ask(label, default=default).strip()
        if value:
            return value


def _confidence_level_from_prior(prior: Dict[str, Any]) -> int:
    label = str(prior.get("confidence_label") or "").strip().lower()
    for level, (candidate, _) in _CONFIDENCE_LEVELS.items():
        if label == candidate:
            return level
    value = prior.get("confidence")
    if value is None:
        return 3
    try:
        numeric = float(value)
    except Exception:
        return 3
    return min(
        _CONFIDENCE_LEVELS,
        key=lambda level: abs(_CONFIDENCE_LEVELS[level][1] - numeric),
    )


def _prompt_confidence_category(console, label: str, prior: Dict[str, Any]) -> Tuple[str, float]:
    default_level = _confidence_level_from_prior(prior)
    console.print("[bold]Confidence levels[/bold]")
    for level, (name, numeric) in _CONFIDENCE_LEVELS.items():
        default_tag = " [dim](default)[/dim]" if level == default_level else ""
        console.print(f"{level}. {name}{default_tag}")
    selected = IntPrompt.ask(f"{label} [1-4]", default=default_level)
    if selected not in _CONFIDENCE_LEVELS:
        selected = default_level
    return _CONFIDENCE_LEVELS[selected]


def _normalize_doc_ref(token: str) -> str:
    cleaned = token.strip().lower()
    if not cleaned:
        return ""
    if cleaned.startswith("d") and cleaned[1:].isdigit():
        return cleaned
    if cleaned.isdigit():
        return f"d{cleaned}"
    return cleaned


def _prompt_doc_refs(console, label: str, valid_doc_ids: List[str], default: str = "", allow_blank: bool = True, exact_count: Optional[int] = None) -> List[str]:
    valid_set = {doc_id.lower() for doc_id in valid_doc_ids}
    while True:
        raw = Prompt.ask(label, default=default).strip()
        if not raw:
            if allow_blank:
                return []
            console.print("[red]Please enter at least one doc id.[/red]")
            continue
        normalized: List[str] = []
        invalid: List[str] = []
        seen: set[str] = set()
        for token in raw.split(","):
            doc_id = _normalize_doc_ref(token)
            if not doc_id:
                continue
            if doc_id not in valid_set:
                invalid.append(token.strip())
                continue
            if doc_id not in seen:
                normalized.append(doc_id)
                seen.add(doc_id)
        if invalid:
            console.print(f"[red]Invalid doc ids: {', '.join(invalid)}. Use only eligible docs such as d1,d2 or 1,2.[/red]")
            continue
        if exact_count is not None and len(normalized) != exact_count:
            console.print(f"[red]Please enter exactly {exact_count} eligible doc ids.[/red]")
            continue
        return normalized


def _save_state_with_event(study_dir, reviewer: str, sample_id: str, status: str, draft: Dict[str, Any]) -> Dict[str, Any]:
    saved = save_judgment(study_dir, reviewer, sample_id, status, draft)
    append_event(study_dir, f"save_{status}", saved)
    return saved


def _show_overview_page(
    console,
    study_dir,
    manifest: Dict[str, Any],
    reviewer: str,
    sample_ids: List[str],
    sample: Dict[str, Any],
    position: int,
    autosave_label: str,
) -> None:
    console.clear()
    progress = get_assignment_progress(study_dir, reviewer)
    render_dashboard(console, manifest["study_name"], reviewer, progress, len(sample_ids))
    render_sample_overview(console, sample, position + 1, len(sample_ids), autosave_label)


def _show_docs_page(console, study_dir, manifest: Dict[str, Any], reviewer: str, sample_ids: List[str], sample: Dict[str, Any]) -> None:
    console.clear()
    progress = get_assignment_progress(study_dir, reviewer)
    render_dashboard(console, manifest["study_name"], reviewer, progress, len(sample_ids))
    render_docs(console, sample)
    console.input("[dim]Press Enter after reviewing docs, then return to the overview or a metric command.[/dim] ")


def _show_behavior_page(console, study_dir, manifest: Dict[str, Any], reviewer: str, sample_ids: List[str], sample: Dict[str, Any]) -> None:
    console.clear()
    progress = get_assignment_progress(study_dir, reviewer)
    render_dashboard(console, manifest["study_name"], reviewer, progress, len(sample_ids))
    render_behavior_guide(console, sample)


def _show_fg_page(console, study_dir, manifest: Dict[str, Any], reviewer: str, sample_ids: List[str], sample: Dict[str, Any]) -> None:
    console.clear()
    progress = get_assignment_progress(study_dir, reviewer)
    render_dashboard(console, manifest["study_name"], reviewer, progress, len(sample_ids))
    render_fg_guide(console, sample)


def _show_str_page(console, study_dir, manifest: Dict[str, Any], reviewer: str, sample_ids: List[str], sample: Dict[str, Any]) -> None:
    console.clear()
    progress = get_assignment_progress(study_dir, reviewer)
    render_dashboard(console, manifest["study_name"], reviewer, progress, len(sample_ids))
    render_str_guide(console, sample)


def _show_review_page(console, study_dir, manifest: Dict[str, Any], reviewer: str, sample_ids: List[str], sample: Dict[str, Any], draft: Dict[str, Any]) -> None:
    console.clear()
    progress = get_assignment_progress(study_dir, reviewer)
    render_dashboard(console, manifest["study_name"], reviewer, progress, len(sample_ids))
    render_review_summary(console, sample, draft)


def edit_behavior(console, sample: Dict[str, Any], draft: Dict[str, Any]) -> None:
    if sample.get("correct_refusal"):
        draft["behavior"] = {
            "adherent": None,
            "confidence": None,
            "rationale": "Skipped: correct refusal.",
            "skipped": "correct_refusal",
        }
        console.print("[green]Behavior skipped because this sample is a correct refusal.[/green]")
        return
    prior = draft.get("behavior") or {}
    adherent = Confirm.ask("Behavior adherent?", default=bool(prior.get("adherent", True)))
    confidence_label, confidence = _prompt_confidence_category(console, "Behavior confidence", prior)
    rationale = _prompt_rationale("Behavior rationale", default=prior.get("rationale", ""))
    draft["behavior"] = {
        "adherent": bool(adherent),
        "confidence": round(float(confidence), 3),
        "confidence_label": confidence_label,
        "rationale": rationale,
    }


def edit_fg(console, sample: Dict[str, Any], draft: Dict[str, Any]) -> None:
    if sample.get("correct_refusal"):
        draft["fg"] = {
            "grounding_ratio": None,
            "supported_claims": 0,
            "total_claims": 0,
            "claim_details": [],
            "skipped": "correct_refusal",
        }
        console.print("[green]FG skipped because this sample is a correct refusal.[/green]")
        return
    claims = sample.get("claims_with_citations") or []
    eligible_docs = sample.get("fg_eligible_docs") or []
    if not claims:
        draft["fg"] = {
            "grounding_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "claim_details": [],
            "reason": "no_extracted_claims",
        }
        console.print("[yellow]No claims were extracted for this answer, so FG is stored as 0 claims reviewed.[/yellow]")
        return
    claim_details: List[Dict[str, Any]] = []
    supported_count = 0
    valid_doc_ids = [str(doc.get("doc_id") or "").lower() for doc in eligible_docs if doc.get("doc_id")]
    for idx, claim in enumerate(claims, start=1):
        render_claim(console, claim, idx, len(claims), eligible_docs)
        doc_hint = ", ".join(doc.get("doc_id") or "" for doc in eligible_docs) or "none"
        supporting_docs = _prompt_doc_refs(
            console,
            f"Supporting eligible docs for this claim? Enter ALL that support it from [{doc_hint}] (use d1,d2 or 1,2), blank for none",
            valid_doc_ids,
            default="",
            allow_blank=True,
        )
        cross_doc_support = False
        cross_doc_combo: List[str] = []
        if not supporting_docs:
            cross_doc_support = Confirm.ask("No single doc selected. Is this supported only by a 2-doc combination?", default=False)
            if cross_doc_support:
                cross_doc_combo = _prompt_doc_refs(
                    console,
                    "Enter exactly two eligible doc ids (d1,d2 or 1,2)",
                    valid_doc_ids,
                    default="",
                    allow_blank=False,
                    exact_count=2,
                )
        note = Prompt.ask("Optional FG note", default="").strip()
        supported, reason = compute_fg_claim_support(
            claim.get("cited_docs") or [],
            supporting_docs,
            cross_doc_support,
            cross_doc_combo,
        )
        if supported:
            supported_count += 1
        claim_details.append({
            "claim": claim.get("text") or "",
            "cited_docs": claim.get("cited_docs") or [],
            "supporting_docs_found": supporting_docs,
            "cross_doc_support": cross_doc_support,
            "cross_doc_combo": cross_doc_combo,
            "citation_check_passed": supported,
            "supported": supported,
            "reason": reason,
            "human_note": note,
        })
    grounding_ratio = supported_count / len(claims) if claims else 0.0
    draft["fg"] = {
        "grounding_ratio": grounding_ratio,
        "supported_claims": supported_count,
        "total_claims": len(claims),
        "evaluable_claims": len(claims),
        "claim_details": claim_details,
    }


def edit_str(console, sample: Dict[str, Any], draft: Dict[str, Any]) -> None:
    if sample.get("correct_refusal"):
        draft["str"] = {
            "adherent": None,
            "confidence": None,
            "rationale": "Skipped: correct refusal.",
            "skipped": "correct_refusal",
            "recall": 0.0,
        }
        console.print("[green]STR skipped because this sample is a correct refusal.[/green]")
        return
    if not sample.get("single_truth_applicable"):
        draft["str"] = {
            "adherent": None,
            "confidence": None,
            "rationale": "Skipped: no gold answer or STR not applicable for this conflict type.",
            "skipped": "not_applicable",
            "recall": 0.0,
        }
        console.print("[yellow]STR is not applicable for this sample.[/yellow]")
        return
    prior = draft.get("str") or {}
    adherent = Confirm.ask("Does the model ASSERT the gold answer?", default=bool(prior.get("adherent", True)))
    confidence_label, confidence = _prompt_confidence_category(console, "STR confidence", prior)
    rationale = _prompt_rationale("STR rationale", default=prior.get("rationale", ""))
    draft["str"] = {
        "adherent": bool(adherent),
        "confidence": round(float(confidence), 3),
        "confidence_label": confidence_label,
        "rationale": rationale,
        "recall": 1.0 if adherent else 0.0,
    }


def edit_notes(draft: Dict[str, Any]) -> None:
    draft.setdefault("meta", {})
    current = draft["meta"].get("notes", "")
    draft["meta"]["notes"] = Prompt.ask("Reviewer notes", default=current)


def _is_complete(sample: Dict[str, Any], draft: Dict[str, Any]) -> bool:
    behavior_ok = bool(draft.get("behavior")) or sample.get("correct_refusal")
    fg_ok = bool(draft.get("fg")) or sample.get("correct_refusal")
    str_ok = bool(draft.get("str")) or (not sample.get("single_truth_applicable")) or sample.get("correct_refusal")
    return bool(behavior_ok and fg_ok and str_ok)


def run_judge_session(study_dir, reviewer: str, sample_ids: List[str], include_submitted: bool = False) -> None:
    console = app_console()
    manifest = load_manifest(study_dir)
    samples = sample_index(study_dir)
    if not sample_ids:
        console.print("[red]No assignments found for this reviewer.[/red]")
        return
    position = 0
    while 0 <= position < len(sample_ids):
        sample = samples[sample_ids[position]]
        latest = get_latest_judgment(study_dir, reviewer, sample["sample_id"])
        if latest and latest["status"] == "submitted" and not include_submitted:
            position += 1
            continue
        draft = _seed_draft(sample, reviewer, latest)
        current_status = latest["status"] if latest else "draft"
        autosave_label = "submitted" if latest and latest["status"] == "submitted" else ("draft" if latest else "new")
        _show_overview_page(console, study_dir, manifest, reviewer, sample_ids, sample, position, autosave_label)
        while True:
            render_help(console)
            command = Prompt.ask("[bold green]Command[/bold green]", default="r").strip().lower()
            if command == "h":
                _show_overview_page(console, study_dir, manifest, reviewer, sample_ids, sample, position, autosave_label)
            elif command == "o":
                _show_overview_page(console, study_dir, manifest, reviewer, sample_ids, sample, position, autosave_label)
            elif command == "d":
                _show_docs_page(console, study_dir, manifest, reviewer, sample_ids, sample)
            elif command == "b":
                _show_behavior_page(console, study_dir, manifest, reviewer, sample_ids, sample)
                edit_behavior(console, sample, draft)
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]Autosaved after behavior edit ({autosave_label}).[/green]")
            elif command == "f":
                _show_fg_page(console, study_dir, manifest, reviewer, sample_ids, sample)
                edit_fg(console, sample, draft)
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]Autosaved after FG edit ({autosave_label}).[/green]")
            elif command == "t":
                _show_str_page(console, study_dir, manifest, reviewer, sample_ids, sample)
                edit_str(console, sample, draft)
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]Autosaved after STR edit ({autosave_label}).[/green]")
            elif command == "m":
                _show_review_page(console, study_dir, manifest, reviewer, sample_ids, sample, draft)
                edit_notes(draft)
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]Autosaved after notes edit ({autosave_label}).[/green]")
            elif command == "r":
                _show_review_page(console, study_dir, manifest, reviewer, sample_ids, sample, draft)
            elif command == "s":
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]State saved ({autosave_label}).[/green]")
            elif command == "x":
                if not _is_complete(sample, draft):
                    console.print("[yellow]This sample is still incomplete. Fill all applicable metrics before submitting.[/yellow]")
                    continue
                saved = save_judgment(study_dir, reviewer, sample["sample_id"], "submitted", draft)
                append_event(study_dir, "submit_judgment", saved)
                current_status = "submitted"
                console.print(f"[bold green]Submitted sample {sample['sample_id']}.[/bold green]")
                position += 1
                break
            elif command == "n":
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]Autosaved before moving next ({autosave_label}).[/green]")
                position += 1
                break
            elif command == "p":
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]Autosaved before moving previous ({autosave_label}).[/green]")
                position = max(0, position - 1)
                break
            elif command == "j":
                saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                autosave_label = f"{saved['status']} r{saved['revision']}"
                console.print(f"[green]Autosaved before jump ({autosave_label}).[/green]")
                jump = IntPrompt.ask("Jump to assigned sample number", default=position + 1)
                position = min(max(jump - 1, 0), len(sample_ids) - 1)
                break
            elif command == "q":
                if Confirm.ask("Save current state before quitting?", default=True):
                    saved = _save_state_with_event(study_dir, reviewer, sample["sample_id"], current_status, draft)
                    console.print(f"[green]State saved ({saved['status']} r{saved['revision']}).[/green]")
                console.print("[bold cyan]Session closed.[/bold cyan]")
                return
            else:
                console.print("[yellow]Unknown command. Press h for help.[/yellow]")

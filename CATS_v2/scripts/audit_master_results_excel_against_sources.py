#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET


ROOT = Path("outputs/benchmark_local_committee_3judge")
MASTER_DIR = ROOT / "master_results"
XLSX_PATH = MASTER_DIR / "Master Results.xlsx"
OUT_JSON = MASTER_DIR / "master_results_excel_source_audit_20260709.json"
OUT_MD = MASTER_DIR / "master_results_excel_source_audit_20260709.md"

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TOL = 1e-12

HEADER_ALIASES = {
    "collection": "collection",
    "category": "category",
    "variant": "variant",
    "model_family": "model_family",
    "eval_family": "eval_family",
    "prompt_mode": "prompt_mode",
    "run_type": "run_type",
    "gr_answer_precision": "gr_answer_precision",
    "gr_answering_precision": "gr_answer_precision",
    "gr_answer_recall": "gr_answer_recall",
    "gr_answering_recall": "gr_answer_recall",
    "gr_answer_f1": "gr_answer_f1",
    "gr_answering_f1": "gr_answer_f1",
    "gr_refusal_precision": "gr_refusal_precision",
    "gr_refusal_recall": "gr_refusal_recall",
    "gr_refusal_f1": "gr_refusal_f1",
    "gr_accuracy": "gr_accuracy",
    "str": "str",
    "fg": "fg",
    "behavior": "behavior",
    "final_cats": "final_cats",
    "n": "n",
    "behavior_n": "behavior_n",
    "fg_n": "fg_n",
    "str_n": "str_n",
    "correct_refusals": "correct_refusals",
}

META_FIELDS = [
    "collection",
    "category",
    "variant",
    "model_family",
    "eval_family",
    "prompt_mode",
    "run_type",
]
NUMERIC_FIELDS = [
    "gr_answer_precision",
    "gr_answer_recall",
    "gr_answer_f1",
    "gr_refusal_precision",
    "gr_refusal_recall",
    "gr_refusal_f1",
    "gr_accuracy",
    "str",
    "fg",
    "behavior",
    "final_cats",
    "n",
    "behavior_n",
    "fg_n",
    "str_n",
]
OPTIONAL_NUMERIC_FIELDS = ["correct_refusals"]


@dataclass(frozen=True)
class SourceRow:
    source_relpath: str
    collection: str
    category: str
    variant: str
    model_family: str
    eval_family: str
    prompt_mode: str
    run_type: str
    gr_answer_precision: float
    gr_answer_recall: float
    gr_answer_f1: float
    gr_refusal_precision: float
    gr_refusal_recall: float
    gr_refusal_f1: float
    gr_accuracy: float
    str_score: float
    fg: float
    behavior: float
    final_cats: float
    n: int
    behavior_n: int
    fg_n: int
    str_n: int
    correct_refusals: int

    def as_expected_map(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "category": self.category,
            "variant": self.variant,
            "model_family": self.model_family,
            "eval_family": self.eval_family,
            "prompt_mode": self.prompt_mode,
            "run_type": self.run_type,
            "gr_answer_precision": self.gr_answer_precision,
            "gr_answer_recall": self.gr_answer_recall,
            "gr_answer_f1": self.gr_answer_f1,
            "gr_refusal_precision": self.gr_refusal_precision,
            "gr_refusal_recall": self.gr_refusal_recall,
            "gr_refusal_f1": self.gr_refusal_f1,
            "gr_accuracy": self.gr_accuracy,
            "str": self.str_score,
            "fg": self.fg,
            "behavior": self.behavior,
            "final_cats": self.final_cats,
            "n": self.n,
            "behavior_n": self.behavior_n,
            "fg_n": self.fg_n,
            "str_n": self.str_n,
            "correct_refusals": self.correct_refusals,
        }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def infer_metadata(rel: Path) -> dict[str, str]:
    parts = rel.parts
    if parts[0] == "benchmark_set_all_modes":
        if parts[1] == "answer_only_sft":
            if len(parts) != 8 or parts[-2:] != ("final", "detailed_results.json"):
                raise ValueError(f"Unexpected answer_only path shape: {rel}")
            return {
                "collection": "benchmark_set_all_modes",
                "category": "answer_only_sft",
                "variant": "answer_only_sft",
                "model_family": parts[2],
                "eval_family": parts[3],
                "prompt_mode": parts[4],
                "run_type": parts[5],
            }
        if len(parts) != 7 or parts[-2:] != ("final", "detailed_results.json"):
            raise ValueError(f"Unexpected standard benchmark path shape: {rel}")
        return {
            "collection": "benchmark_set_all_modes",
            "category": "standard_benchmark",
            "variant": "standard",
            "model_family": parts[1],
            "eval_family": parts[2],
            "prompt_mode": parts[3],
            "run_type": parts[4],
        }
    if parts[0] == "other_techniques":
        if len(parts) != 5 or parts[-2:] != ("final", "detailed_results.json"):
            raise ValueError(f"Unexpected other-techniques path shape: {rel}")
        return {
            "collection": "other_techniques",
            "category": "other_techniques",
            "variant": parts[1],
            "model_family": parts[2],
            "eval_family": parts[1],
            "prompt_mode": "comparison",
            "run_type": "committee_eval",
        }
    raise ValueError(f"Unexpected result root: {rel}")


def load_source_rows() -> list[SourceRow]:
    rows: list[SourceRow] = []
    for path in sorted(ROOT.rglob("detailed_results.json")):
        if path.parent == MASTER_DIR:
            continue
        rel = path.relative_to(ROOT)
        data = load_json(path)
        summary = data["summary"]
        overall = summary["conflict_overall"]
        gr = summary["gr_dataset_metrics"]
        meta = infer_metadata(rel)
        rows.append(
            SourceRow(
                source_relpath=str(rel),
                collection=meta["collection"],
                category=meta["category"],
                variant=meta["variant"],
                model_family=meta["model_family"],
                eval_family=meta["eval_family"],
                prompt_mode=meta["prompt_mode"],
                run_type=meta["run_type"],
                gr_answer_precision=gr["precision"],
                gr_answer_recall=gr["recall"],
                gr_answer_f1=gr["f1"],
                gr_refusal_precision=gr["abstain_precision"],
                gr_refusal_recall=gr["abstain_recall"],
                gr_refusal_f1=gr["abstain_f1"],
                gr_accuracy=overall["gr_accuracy"],
                str_score=overall["single_truth_recall"],
                fg=overall["factual_grounding"],
                behavior=overall["behavior"],
                final_cats=overall["cats_score"],
                n=overall["n"],
                behavior_n=overall["behavior_n"],
                fg_n=overall["factual_grounding_n"],
                str_n=overall["single_truth_recall_n"],
                correct_refusals=overall["correct_refusals"],
            )
        )
    return rows


def sort_key(row: SourceRow) -> tuple[Any, ...]:
    return (
        row.collection,
        row.category,
        row.model_family,
        row.eval_family,
        row.prompt_mode,
        row.run_type,
        row.source_relpath,
    )


def load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in root.findall("a:si", NS):
        out.append("".join((t.text or "") for t in si.iterfind(".//a:t", NS)))
    return out


def cell_value(cell: ET.Element, sst: list[str]) -> str:
    kind = cell.attrib.get("t")
    value = cell.find("a:v", NS)
    inline = cell.find("a:is", NS)
    if kind == "s" and value is not None:
        return sst[int(value.text)]
    if kind == "inlineStr" and inline is not None:
        return "".join((t.text or "") for t in inline.iterfind(".//a:t", NS))
    return "" if value is None else (value.text or "")


def col_letters(ref: str) -> str:
    return "".join(ch for ch in ref if ch.isalpha())


def load_workbook_rows() -> tuple[list[str], list[dict[str, str]], dict[str, str]]:
    with ZipFile(XLSX_PATH) as zf:
        sst = load_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = sheet.find("a:sheetData", NS)
        if rows is None:
            raise ValueError("sheetData missing in workbook")

        parsed_rows: list[tuple[int, dict[str, str]]] = []
        for row in rows.findall("a:row", NS):
            rid = int(row.attrib["r"])
            raw: dict[str, str] = {}
            for cell in row.findall("a:c", NS):
                raw[col_letters(cell.attrib["r"])] = cell_value(cell, sst).strip()
            parsed_rows.append((rid, raw))

    header_row = next(raw for rid, raw in parsed_rows if rid == 1)
    col_to_field: dict[str, str] = {}
    duplicate_headers: dict[str, list[str]] = {}
    unrecognized_headers: dict[str, str] = {}
    for col, text in sorted(header_row.items()):
        if not text:
            continue
        canonical = HEADER_ALIASES.get(text)
        if canonical is None:
            unrecognized_headers[col] = text
            continue
        if canonical in col_to_field.values():
            duplicate_headers.setdefault(canonical, []).append(col)
        col_to_field[col] = canonical

    normalized: list[dict[str, str]] = []
    previous: dict[str, str] = {field: "" for field in META_FIELDS}
    used_columns = set(col_to_field)
    for rid, raw in parsed_rows:
        if rid == 1:
            continue
        if not any((raw.get(col, "") or "").strip() for col in used_columns):
            continue
        row_values: dict[str, str] = {"excel_row": str(rid)}
        for col, field in col_to_field.items():
            value = (raw.get(col, "") or "").strip()
            if field in META_FIELDS and value == "":
                value = previous[field]
            row_values[field] = value
        for field in META_FIELDS:
            previous[field] = row_values.get(field, previous[field])
        normalized.append(row_values)

    header_info = {
        "recognized_column_count": str(len(col_to_field)),
        "unrecognized_headers": json.dumps(unrecognized_headers, sort_keys=True),
        "duplicate_headers": json.dumps(duplicate_headers, sort_keys=True),
    }
    ordered_headers = [header_row[col] for col in sorted(col_to_field)]
    return ordered_headers, normalized, header_info


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def as_int_text(value: int) -> str:
    return str(int(value))


def numeric_fields_match(workbook: dict[str, str], expected_map: dict[str, Any]) -> bool:
    for field in NUMERIC_FIELDS:
        actual_text = workbook.get(field, "")
        if actual_text == "":
            return False
        if not close(float(actual_text), float(expected_map[field])):
            return False
    for field in OPTIONAL_NUMERIC_FIELDS:
        if field in workbook and workbook[field] != "":
            if not close(float(workbook[field]), float(expected_map[field])):
                return False
    return True


def audit() -> dict[str, Any]:
    source_rows = sorted(load_source_rows(), key=sort_key)
    _, workbook_rows, header_info = load_workbook_rows()

    row_count_mismatch = len(workbook_rows) != len(source_rows)
    source_by_key: dict[tuple[Any, ...], SourceRow] = {}
    duplicate_source_keys: list[dict[str, Any]] = []
    for row in source_rows:
        if row.category == "other_techniques":
            key = (row.collection, row.category, row.variant, row.model_family, row.run_type)
        else:
            key = (row.collection, row.category, row.model_family, row.eval_family, row.prompt_mode, row.run_type)
        if key in source_by_key:
            duplicate_source_keys.append({"key": key, "existing": source_by_key[key].source_relpath, "duplicate": row.source_relpath})
        source_by_key[key] = row

    row_issues: list[dict[str, Any]] = []
    metadata_only_issues: list[dict[str, Any]] = []
    checked_numeric_cells = 0
    checked_metadata_cells = 0
    matched_source_relpaths: set[str] = set()

    for workbook in workbook_rows:
        if workbook["category"] == "other_techniques":
            key = (
                workbook.get("collection", ""),
                workbook.get("category", ""),
                workbook.get("variant", ""),
                workbook.get("model_family", ""),
                workbook.get("run_type", ""),
            )
        else:
            key = (
                workbook.get("collection", ""),
                workbook.get("category", ""),
                workbook.get("model_family", ""),
                workbook.get("eval_family", ""),
                workbook.get("prompt_mode", ""),
                workbook.get("run_type", ""),
            )

        expected = source_by_key.get(key)
        if expected is None:
            numeric_matches = [
                row for row in source_rows if numeric_fields_match(workbook, row.as_expected_map())
            ]
            if len(numeric_matches) == 1:
                expected = numeric_matches[0]
                metadata_only_issues.append(
                    {
                        "excel_row": int(workbook["excel_row"]),
                        "source_relpath": expected.source_relpath,
                        "issue": "metadata_key_mismatch_but_numeric_values_match_unique_source_row",
                        "workbook_key": key,
                    }
                )
            else:
                row_issues.append(
                    {
                        "excel_row": int(workbook["excel_row"]),
                        "issue": "no_matching_source_row_for_metadata_key",
                        "workbook_key": key,
                        "numeric_match_count": len(numeric_matches),
                        "numeric_match_relpaths": [row.source_relpath for row in numeric_matches[:10]],
                    }
                )
                continue

        matched_source_relpaths.add(expected.source_relpath)
        expected_map = expected.as_expected_map()
        mismatches: dict[str, Any] = {}

        for field in META_FIELDS:
            checked_metadata_cells += 1
            actual = workbook.get(field, "")
            wanted = str(expected_map[field])
            if actual != wanted:
                mismatches[field] = {"workbook": actual, "expected": wanted}

        for field in NUMERIC_FIELDS:
            checked_numeric_cells += 1
            actual_text = workbook.get(field, "")
            if actual_text == "":
                mismatches[field] = {"workbook": actual_text, "expected": expected_map[field]}
                continue
            actual = float(actual_text)
            wanted = float(expected_map[field])
            if not close(actual, wanted):
                mismatches[field] = {"workbook": actual, "expected": wanted}

        for field in OPTIONAL_NUMERIC_FIELDS:
            if field in workbook and workbook[field] != "":
                checked_numeric_cells += 1
                actual = float(workbook[field])
                wanted = float(expected_map[field])
                if not close(actual, wanted):
                    mismatches[field] = {"workbook": actual, "expected": wanted}

        if mismatches:
            numeric_mismatch_fields = set(NUMERIC_FIELDS + OPTIONAL_NUMERIC_FIELDS) & set(mismatches)
            target = row_issues if numeric_mismatch_fields else metadata_only_issues
            target.append(
                {
                    "excel_row": int(workbook["excel_row"]),
                    "source_relpath": expected.source_relpath,
                    "mismatches": mismatches,
                }
            )

    missing_source_rows = sorted(
        row.source_relpath for row in source_rows if row.source_relpath not in matched_source_relpaths
    )
    for rel in missing_source_rows:
        row_issues.append({"source_relpath": rel, "issue": "source_row_missing_in_workbook"})

    if len(workbook_rows) > len(source_rows):
        for workbook in workbook_rows[len(source_rows):]:
            row_issues.append(
                {
                    "excel_row": int(workbook["excel_row"]),
                    "issue": "extra_workbook_row",
                }
            )

    metrics_present = sorted(
        field for field in NUMERIC_FIELDS + OPTIONAL_NUMERIC_FIELDS if any(field in row for row in workbook_rows)
    )

    report = {
        "workbook_path": str(XLSX_PATH),
        "verified_source_result_count": len(source_rows),
        "workbook_data_row_count": len(workbook_rows),
        "row_count_matches": not row_count_mismatch,
        "checked_metadata_cells": checked_metadata_cells,
        "checked_numeric_cells": checked_numeric_cells,
        "metrics_present_in_workbook": metrics_present,
        "header_info": header_info,
        "duplicate_source_key_count": len(duplicate_source_keys),
        "duplicate_source_keys_preview": duplicate_source_keys[:20],
        "missing_source_row_count": len(missing_source_rows),
        "missing_source_rows_preview": missing_source_rows[:20],
        "metadata_only_issue_count": len(metadata_only_issues),
        "metadata_only_issues_preview": metadata_only_issues[:20],
        "issue_count": len(row_issues),
        "issues_preview": row_issues[:50],
        "numeric_values_ok": len(row_issues) == 0,
        "overall_ok": (
            not row_count_mismatch
            and len(duplicate_source_keys) == 0
            and len(missing_source_rows) == 0
            and len(row_issues) == 0
        ),
    }
    return report


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Master Results Excel Source Audit",
        "",
        f"- Verified source result files: `{report['verified_source_result_count']}`",
        f"- Workbook data rows detected: `{report['workbook_data_row_count']}`",
        f"- Row count matches source count: `{report['row_count_matches']}`",
        f"- Checked metadata cells: `{report['checked_metadata_cells']}`",
        f"- Checked numeric cells: `{report['checked_numeric_cells']}`",
        f"- Metrics present in workbook: `{', '.join(report['metrics_present_in_workbook'])}`",
        f"- Issue count: `{report['issue_count']}`",
        f"- Overall OK: `{report['overall_ok']}`",
        "",
        "This audit compares the Excel workbook directly against the synced source",
        "`detailed_results.json` files under `outputs/benchmark_local_committee_3judge`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    report = audit()
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

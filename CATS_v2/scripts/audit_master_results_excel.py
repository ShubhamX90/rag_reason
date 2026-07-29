#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET


MASTER_DIR = Path("outputs/benchmark_local_committee_3judge/master_results")
XLSX_PATH = MASTER_DIR / "Master Results.xlsx"
CSV_PATH = MASTER_DIR / "cats_master_results_20260708.csv"
OUT_JSON = MASTER_DIR / "master_results_excel_audit_20260709.json"
OUT_MD = MASTER_DIR / "master_results_excel_audit_20260709.md"

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
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
    "gr_f1",
    "gr_precision",
    "gr_recall",
    "gr_accuracy",
    "str",
    "fg",
    "behavior",
    "final_cats",
    "n",
    "behavior_n",
    "fg_n",
    "str_n",
    "correct_refusals",
]
ALL_FIELDS = META_FIELDS + NUMERIC_FIELDS
FIELD_TO_COL = {field: chr(ord("A") + idx) for idx, field in enumerate(ALL_FIELDS)}


@dataclass
class WorkbookRow:
    excel_row: int
    values: dict[str, str]


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


def load_workbook_rows() -> list[WorkbookRow]:
    with ZipFile(XLSX_PATH) as zf:
        sst = load_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = sheet.find("a:sheetData", NS)
        if rows is None:
            raise ValueError("sheetData missing in workbook")

        all_rows: list[tuple[int, dict[str, str]]] = []
        for row in rows.findall("a:row", NS):
            rid = int(row.attrib["r"])
            raw: dict[str, str] = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib["r"]
                col = "".join(ch for ch in ref if ch.isalpha())
                raw[col] = cell_value(cell, sst)
            all_rows.append((rid, raw))

    # Remove header and completely blank spacer rows.
    data_rows = [
        (rid, raw)
        for rid, raw in all_rows
        if rid != 1 and any((raw.get(col, "") or "").strip() for col in FIELD_TO_COL.values())
    ]

    normalized: list[WorkbookRow] = []
    previous: dict[str, str] = {field: "" for field in ALL_FIELDS}
    for rid, raw in data_rows:
        current: dict[str, str] = {}
        for field in ALL_FIELDS:
            col = FIELD_TO_COL[field]
            value = (raw.get(col, "") or "").strip()
            if field in META_FIELDS and value == "":
                value = previous[field]
            current[field] = value
        previous = current.copy()
        normalized.append(WorkbookRow(excel_row=rid, values=current))
    return normalized


def load_csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def floatish(value: str) -> float:
    return float(value)


def numeric_signature_matches(workbook_row: dict[str, str], csv_row: dict[str, str]) -> bool:
    for field in NUMERIC_FIELDS:
        a = floatish(workbook_row[field])
        b = floatish(csv_row[field])
        if field in {"n", "behavior_n", "fg_n", "str_n", "correct_refusals"}:
            if int(round(a)) != int(round(b)):
                return False
        else:
            if abs(a - b) > 1e-12:
                return False
    return True


def audit() -> dict:
    workbook_rows = load_workbook_rows()
    csv_rows = load_csv_rows()

    issues: list[dict] = []
    unmatched_rows: list[dict] = []
    matched_csv_relpaths: set[str] = set()

    for row in workbook_rows:
        matches = [c for c in csv_rows if numeric_signature_matches(row.values, c)]
        if len(matches) != 1:
            unmatched_rows.append(
                {
                    "excel_row": row.excel_row,
                    "match_count": len(matches),
                    "workbook_values": row.values,
                    "matched_source_relpaths": [m["source_relpath"] for m in matches[:10]],
                }
            )
            continue

        canonical = matches[0]
        matched_csv_relpaths.add(canonical["source_relpath"])
        metadata_mismatches = {}
        for field in META_FIELDS:
            workbook_value = row.values[field]
            csv_value = canonical[field]
            if workbook_value != csv_value:
                metadata_mismatches[field] = {
                    "workbook": workbook_value,
                    "canonical": csv_value,
                }

        if metadata_mismatches:
            issues.append(
                {
                    "excel_row": row.excel_row,
                    "source_relpath": canonical["source_relpath"],
                    "metadata_mismatches": metadata_mismatches,
                }
            )

    missing_csv_rows = sorted(
        row["source_relpath"] for row in csv_rows if row["source_relpath"] not in matched_csv_relpaths
    )

    report = {
        "workbook_path": str(XLSX_PATH),
        "csv_path": str(CSV_PATH),
        "workbook_data_row_count": len(workbook_rows),
        "csv_row_count": len(csv_rows),
        "unmatched_workbook_rows": unmatched_rows,
        "metadata_issue_count": len(issues),
        "metadata_issues": issues,
        "missing_csv_rows_after_numeric_matching": missing_csv_rows,
        "numeric_values_ok_for_matched_rows": len(unmatched_rows) == 0,
        "overall_ok": len(unmatched_rows) == 0 and len(issues) == 0 and not missing_csv_rows,
    }
    return report


def write_markdown(report: dict) -> None:
    lines = [
        "# Master Results Excel Audit",
        "",
        f"- Workbook data rows detected: `{report['workbook_data_row_count']}`",
        f"- Canonical CSV rows: `{report['csv_row_count']}`",
        f"- Numerically unmatched workbook rows: `{len(report['unmatched_workbook_rows'])}`",
        f"- Metadata issue count: `{report['metadata_issue_count']}`",
        f"- Overall OK: `{report['overall_ok']}`",
        "",
    ]

    if report["metadata_issues"]:
        lines.append("## Metadata Issues")
        lines.append("")
        for issue in report["metadata_issues"][:30]:
            lines.append(f"- Excel row `{issue['excel_row']}` -> `{issue['source_relpath']}`")
            for field, values in issue["metadata_mismatches"].items():
                lines.append(
                    f"  `{field}` workbook=`{values['workbook']}` canonical=`{values['canonical']}`"
                )
        lines.append("")

    if report["unmatched_workbook_rows"]:
        lines.append("## Unmatched Workbook Rows")
        lines.append("")
        for issue in report["unmatched_workbook_rows"][:10]:
            lines.append(
                f"- Excel row `{issue['excel_row']}` had `{issue['match_count']}` numeric matches"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    report = audit()
    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

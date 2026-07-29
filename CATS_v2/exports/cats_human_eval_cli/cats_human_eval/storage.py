from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def db_path(study_dir: Path) -> Path:
    return study_dir / "state" / "judgments.sqlite3"


def events_path(study_dir: Path) -> Path:
    return study_dir / "state" / "events.jsonl"


def connect_db(study_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(study_dir))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(study_dir: Path) -> None:
    conn = connect_db(study_dir)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reviewers (
              reviewer_id TEXT PRIMARY KEY,
              display_name TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assignments (
              reviewer_id TEXT NOT NULL,
              sample_id TEXT NOT NULL,
              ordinal INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (reviewer_id, sample_id)
            );

            CREATE TABLE IF NOT EXISTS judgments (
              judgment_id INTEGER PRIMARY KEY AUTOINCREMENT,
              reviewer_id TEXT NOT NULL,
              sample_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              submitted_at TEXT,
              payload_json TEXT NOT NULL,
              active INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def append_event(study_dir: Path, event_type: str, payload: Dict[str, Any]) -> None:
    row = {
        "timestamp": utc_now_iso(),
        "event_type": event_type,
        "payload": payload,
    }
    with events_path(study_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_reviewers(study_dir: Path, reviewers: Iterable[str]) -> None:
    now = utc_now_iso()
    conn = connect_db(study_dir)
    try:
        for reviewer in reviewers:
            conn.execute(
                """
                INSERT INTO reviewers (reviewer_id, display_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(reviewer_id) DO UPDATE SET display_name=excluded.display_name
                """,
                (reviewer, reviewer, now),
            )
        conn.commit()
    finally:
        conn.close()


def replace_assignments(
    study_dir: Path,
    reviewer_to_samples: Dict[str, List[str]],
) -> None:
    conn = connect_db(study_dir)
    now = utc_now_iso()
    try:
        conn.execute("DELETE FROM assignments")
        for reviewer, sample_ids in reviewer_to_samples.items():
            for ordinal, sample_id in enumerate(sample_ids):
                conn.execute(
                    """
                    INSERT INTO assignments (reviewer_id, sample_id, ordinal, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (reviewer, sample_id, ordinal, now),
                )
        conn.commit()
    finally:
        conn.close()


def get_assignments(study_dir: Path, reviewer: str) -> List[str]:
    conn = connect_db(study_dir)
    try:
        rows = conn.execute(
            """
            SELECT sample_id
            FROM assignments
            WHERE reviewer_id = ?
            ORDER BY ordinal ASC
            """,
            (reviewer,),
        ).fetchall()
        return [row["sample_id"] for row in rows]
    finally:
        conn.close()


def get_assignment_progress(study_dir: Path, reviewer: str) -> Dict[str, int]:
    conn = connect_db(study_dir)
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM assignments WHERE reviewer_id = ?",
            (reviewer,),
        ).fetchone()["n"]
        submitted = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM assignments a
            WHERE reviewer_id = ?
              AND EXISTS (
                SELECT 1
                FROM judgments j
                WHERE j.reviewer_id = a.reviewer_id
                  AND j.sample_id = a.sample_id
                  AND j.active = 1
                  AND j.status = 'submitted'
              )
            """,
            (reviewer,),
        ).fetchone()["n"]
        drafts = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM assignments a
            WHERE reviewer_id = ?
              AND EXISTS (
                SELECT 1
                FROM judgments j
                WHERE j.reviewer_id = a.reviewer_id
                  AND j.sample_id = a.sample_id
                  AND j.active = 1
                  AND j.status = 'draft'
              )
            """,
            (reviewer,),
        ).fetchone()["n"]
        return {"total": total, "submitted": submitted, "drafts": drafts}
    finally:
        conn.close()


def get_latest_judgment(
    study_dir: Path,
    reviewer: str,
    sample_id: str,
) -> Optional[Dict[str, Any]]:
    conn = connect_db(study_dir)
    try:
        row = conn.execute(
            """
            SELECT *
            FROM judgments
            WHERE reviewer_id = ? AND sample_id = ? AND active = 1
            ORDER BY revision DESC
            LIMIT 1
            """,
            (reviewer, sample_id),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["payload"] = json.loads(data.pop("payload_json"))
        return data
    finally:
        conn.close()


def save_judgment(
    study_dir: Path,
    reviewer: str,
    sample_id: str,
    status: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    conn = connect_db(study_dir)
    now = utc_now_iso()
    try:
        latest = conn.execute(
            """
            SELECT judgment_id, revision
            FROM judgments
            WHERE reviewer_id = ? AND sample_id = ? AND active = 1
            ORDER BY revision DESC
            LIMIT 1
            """,
            (reviewer, sample_id),
        ).fetchone()
        revision = 1 if latest is None else int(latest["revision"]) + 1
        if latest is not None:
            conn.execute(
                "UPDATE judgments SET active = 0 WHERE judgment_id = ?",
                (latest["judgment_id"],),
            )
        conn.execute(
            """
            INSERT INTO judgments (
              reviewer_id, sample_id, revision, status, created_at, submitted_at, payload_json, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                reviewer,
                sample_id,
                revision,
                status,
                now,
                now if status == "submitted" else None,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()
        return {"reviewer_id": reviewer, "sample_id": sample_id, "revision": revision, "status": status}
    finally:
        conn.close()


def list_active_judgments(study_dir: Path) -> List[Dict[str, Any]]:
    conn = connect_db(study_dir)
    try:
        rows = conn.execute(
            """
            SELECT reviewer_id, sample_id, revision, status, created_at, submitted_at, payload_json
            FROM judgments
            WHERE active = 1
            ORDER BY reviewer_id, sample_id
            """
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data.pop("payload_json"))
            out.append(data)
        return out
    finally:
        conn.close()

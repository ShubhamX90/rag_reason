#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "STOPPED",
    "SUSPENDED",
    "TIMEOUT",
}
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "STAGE_OUT"}
MAX_ALLOWED_ALL_FAILED = 5


@dataclass
class Placement:
    partition: str
    qos: str
    request_cpus: int
    reason: str


@dataclass
class StageState:
    name: str
    kind: str
    judge_name: str
    base_url: str
    sbatch_path: str
    cpus: int
    job_id: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    state: str = "not_submitted"
    attempts: int = 0
    last_reason: str = ""
    last_submit_at: Optional[float] = None
    last_update_at: Optional[float] = None
    completed_at: Optional[float] = None


def run_cmd(cmd: list[str], *, env: Optional[dict[str, str]] = None, check: bool = True) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def now_ts() -> float:
    return time.time()


def print_status(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    user_name = os.environ.get("USER") or run_cmd(["whoami"])
    parser = argparse.ArgumentParser(description="Dynamically watch and place benchmark file pipeline jobs.")
    parser.add_argument("input_file", help="Absolute path to benchmark input.jsonl")
    parser.add_argument("--input-root", default=str(root_dir / "inputs" / "prepped_model_eval_inputs" / "benchmark_set_all_modes"))
    parser.add_argument("--gold-file", default=str(root_dir / "data" / "benchmark" / "benchmark_final_v2_holdout_clean_736.jsonl"))
    parser.add_argument("--expected-rows", type=int, default=736)
    parser.add_argument("--output-root", default=f"/scratch/{user_name}/rag-reason/cats_outputs/benchmark_local_committee_3judge")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--qwen-base-url", required=True)
    parser.add_argument("--mistral-base-url", required=True)
    parser.add_argument("--deepseek-base-url", required=True)
    return parser.parse_args()


def choose_placement(root_dir: Path, cpus: int, args: argparse.Namespace) -> Placement:
    output = run_cmd(["bash", str(root_dir / "scripts" / "select_controller_partition.sh"), str(cpus)])
    data: Dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return Placement(
        partition=data["partition"],
        qos=data.get("qos", ""),
        request_cpus=int(data.get("request_cpus", cpus)),
        reason=data.get("reason", ""),
    )


def query_job(job_id: str) -> tuple[str, str]:
    sq = run_cmd(["squeue", "-h", "-j", job_id, "-o", "%T|%P"], check=False)
    if sq:
        state, partition = sq.split("|", 1)
        return state.upper(), partition.strip()

    sacct = run_cmd(
        ["sacct", "-X", "-j", job_id, "--parsable2", "--format=JobIDRaw,State,Partition", "-n"],
        check=False,
    )
    for line in sacct.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id_raw, state, partition = parts[:3]
        if job_id_raw == job_id:
            normalized = state.strip().upper().rstrip("+")
            return normalized, partition.strip()
    return "UNKNOWN", ""


def build_file_root(args: argparse.Namespace) -> Path:
    rel_dir = Path(args.input_file).resolve().parent.relative_to(Path(args.input_root).resolve())
    if args.run_label:
        return Path(args.output_root) / args.run_label / rel_dir
    return Path(args.output_root) / rel_dir


def _results_file_for_stage(stage: StageState, args: argparse.Namespace) -> Path:
    file_root = build_file_root(args)
    if stage.kind == "collect":
        run_dir = file_root / "staged" / f"{stage.judge_name}_collect"
    else:
        run_dir = file_root / "final"
    return run_dir / "detailed_results.json"


def _stage_results_valid(results_path: Path) -> bool:
    try:
        payload = json.loads(results_path.read_text())
    except Exception:
        return False

    per_sample = payload.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        return False

    summary = payload.get("summary", {}).get("conflict_overall", {})
    if not isinstance(summary, dict):
        return False

    behavior = float(summary.get("behavior", 0.0) or 0.0)
    factual_grounding = float(summary.get("factual_grounding", 0.0) or 0.0)
    single_truth = float(summary.get("single_truth_recall", 0.0) or 0.0)

    if behavior == 0.0 and factual_grounding == 0.0 and single_truth == 0.0:
        return False

    behavior_rows = [row for row in per_sample if row.get("behavior_applicable")]
    all_failed = sum(1 for row in behavior_rows if row.get("behavior_details", {}).get("all_failed"))
    if all_failed > MAX_ALLOWED_ALL_FAILED:
        return False

    return True


def stage_artifacts_complete(stage: StageState, args: argparse.Namespace) -> bool:
    file_root = build_file_root(args)
    if stage.kind == "collect":
        run_dir = file_root / "staged" / f"{stage.judge_name}_collect"
    else:
        run_dir = file_root / "final"

    required = [
        run_dir / "run_config.yaml",
        run_dir / "eval_report.md",
        run_dir / "detailed_results.json",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False

    return _stage_results_valid(run_dir / "detailed_results.json")


def stage_artifacts_present(stage: StageState, args: argparse.Namespace) -> bool:
    file_root = build_file_root(args)
    if stage.kind == "collect":
        run_dir = file_root / "staged" / f"{stage.judge_name}_collect"
    else:
        run_dir = file_root / "final"

    required = [
        run_dir / "run_config.yaml",
        run_dir / "eval_report.md",
        run_dir / "detailed_results.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def validate_input(root_dir: Path, args: argparse.Namespace) -> None:
    run_cmd(
        [
            sys.executable,
            str(root_dir / "scripts" / "validate_eval_input_jsonl.py"),
            "--input",
            args.input_file,
            "--mode",
            "benchmark_prepped",
            "--gold",
            args.gold_file,
            "--expected-rows",
            str(args.expected_rows),
        ]
    )


def build_export(args: argparse.Namespace, root_dir: Path, judge_name: str, base_url: str) -> str:
    pairs = {
        "REPO_ROOT": str(root_dir),
        "INPUT_ROOT": args.input_root,
        "INPUT_FILE": args.input_file,
        "JUDGE_NAME": judge_name,
        "BASE_URL": base_url,
        "GOLD_FILE": args.gold_file,
        "EXPECTED_ROWS": str(args.expected_rows),
        "OUTPUT_ROOT": args.output_root,
        "RUN_LABEL": args.run_label,
    }
    return "ALL," + ",".join(f"{k}={v}" for k, v in pairs.items())


def build_final_export(args: argparse.Namespace, root_dir: Path) -> str:
    pairs = {
        "REPO_ROOT": str(root_dir),
        "INPUT_ROOT": args.input_root,
        "INPUT_FILE": args.input_file,
        "QWEN_BASE_URL": args.qwen_base_url,
        "MISTRAL_BASE_URL": args.mistral_base_url,
        "DEEPSEEK_BASE_URL": args.deepseek_base_url,
        "OUTPUT_ROOT": args.output_root,
        "RUN_LABEL": args.run_label,
    }
    return "ALL," + ",".join(f"{k}={v}" for k, v in pairs.items())


def submit_stage(stage: StageState, placement: Placement, export_spec: str) -> str:
    cmd = ["sbatch", "--parsable", "--partition", placement.partition]
    if placement.qos:
        cmd.extend(["--qos", placement.qos])
    cmd.extend(["--cpus-per-task", str(placement.request_cpus)])
    cmd.extend(["--export", export_spec, stage.sbatch_path])
    return run_cmd(cmd)


def cancel_job(job_id: str) -> None:
    run_cmd(["scancel", job_id], check=False)


def build_state_file(args: argparse.Namespace) -> Path:
    rel_dir = Path(args.input_file).resolve().parent.relative_to(Path(args.input_root).resolve())
    file_root = Path(args.output_root) / args.run_label / rel_dir if args.run_label else Path(args.output_root) / rel_dir
    file_root.mkdir(parents=True, exist_ok=True)
    return file_root / "watch_state.json"


def _stage_from_payload(payload: Dict[str, Any]) -> StageState:
    return StageState(
        name=payload["name"],
        kind=payload["kind"],
        judge_name=payload["judge_name"],
        base_url=payload["base_url"],
        sbatch_path=payload["sbatch_path"],
        cpus=int(payload["cpus"]),
        job_id=payload.get("job_id"),
        partition=payload.get("partition"),
        qos=payload.get("qos"),
        state=payload.get("state", "not_submitted"),
        attempts=int(payload.get("attempts", 0)),
        last_reason=payload.get("last_reason", ""),
        last_submit_at=payload.get("last_submit_at"),
        last_update_at=payload.get("last_update_at"),
        completed_at=payload.get("completed_at"),
    )


def load_state(path: Path, args: argparse.Namespace) -> Optional[Dict[str, StageState]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("input_file") != args.input_file or payload.get("run_label", "") != args.run_label:
        raise RuntimeError(
            f"Refusing to reuse watch state from {path}: "
            f"input/run_label mismatch ({payload.get('input_file')} / {payload.get('run_label', '')})"
        )
    stages_payload = payload.get("stages", {})
    required = {"qwen", "mistral", "deepseek", "merge"}
    if set(stages_payload) != required:
        raise RuntimeError(f"Refusing to reuse watch state from {path}: unexpected stage keys {sorted(stages_payload)}")
    return {name: _stage_from_payload(stage_payload) for name, stage_payload in stages_payload.items()}


def save_state(path: Path, stages: Dict[str, StageState], args: argparse.Namespace) -> None:
    payload = {
        "input_file": args.input_file,
        "run_label": args.run_label,
        "poll_interval": args.poll_interval,
        "stages": {name: asdict(stage) for name, stage in stages.items()},
        "saved_at": now_ts(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def stage_done(stage: StageState) -> bool:
    return stage.state in TERMINAL_SUCCESS


def stage_failed(stage: StageState) -> bool:
    return stage.state in TERMINAL_FAILURE


def refresh_stage(stage: StageState, args: argparse.Namespace) -> None:
    if not stage.job_id:
        return
    state, partition = query_job(stage.job_id)
    artifacts_complete = stage_artifacts_complete(stage, args)
    artifacts_present = stage_artifacts_present(stage, args)
    if state == "UNKNOWN" and artifacts_complete:
        state = "COMPLETED"
    elif state == "UNKNOWN" and artifacts_present:
        state = "FAILED"
        stage.last_reason = "invalid_artifacts"
    elif state in TERMINAL_SUCCESS and not artifacts_complete:
        state = "FAILED"
        stage.last_reason = "invalid_artifacts"
    stage.state = state
    if partition:
        stage.partition = partition
    stage.last_update_at = now_ts()
    if state in TERMINAL_SUCCESS and stage.completed_at is None:
        stage.completed_at = now_ts()


def maybe_submit_collect(stage: StageState, args: argparse.Namespace, root_dir: Path) -> bool:
    placement = choose_placement(root_dir, stage.cpus, args)
    export_spec = build_export(args, root_dir, stage.judge_name, stage.base_url)

    if stage.job_id and stage.state in ACTIVE_STATES:
        return False

    if stage.job_id and stage.state == "UNKNOWN":
        return False

    if stage.job_id and stage_done(stage):
        return False

    if stage.job_id and stage_failed(stage):
        if stage.attempts >= args.max_retries:
            raise RuntimeError(f"{stage.name} failed permanently after {stage.attempts} attempts")
        print_status(f"[{stage.name}] retrying after failure state={stage.state}")
        stage.job_id = None

    job_id = submit_stage(stage, placement, export_spec)
    stage.job_id = job_id
    stage.partition = placement.partition
    stage.qos = placement.qos
    stage.state = "PENDING"
    stage.last_reason = placement.reason
    stage.last_submit_at = now_ts()
    stage.last_update_at = now_ts()
    stage.attempts += 1
    print_status(
        f"[{stage.name}] submitted job_id={job_id} partition={placement.partition} "
        f"qos={placement.qos or 'none'} cpus={placement.request_cpus} reason={placement.reason}"
    )
    return True


def maybe_submit_merge(stage: StageState, args: argparse.Namespace, root_dir: Path) -> bool:
    placement = choose_placement(root_dir, stage.cpus, args)
    export_spec = build_final_export(args, root_dir)

    if stage.job_id and stage.state in ACTIVE_STATES:
        return False

    if stage.job_id and stage.state == "UNKNOWN":
        return False

    if stage.job_id and stage_done(stage):
        return False

    if stage.job_id and stage_failed(stage):
        if stage.attempts >= args.max_retries:
            raise RuntimeError(f"{stage.name} failed permanently after {stage.attempts} attempts")
        print_status(f"[{stage.name}] retrying after failure state={stage.state}")
        stage.job_id = None

    job_id = submit_stage(stage, placement, export_spec)
    stage.job_id = job_id
    stage.partition = placement.partition
    stage.qos = placement.qos
    stage.state = "PENDING"
    stage.last_reason = placement.reason
    stage.last_submit_at = now_ts()
    stage.last_update_at = now_ts()
    stage.attempts += 1
    print_status(
        f"[{stage.name}] submitted job_id={job_id} partition={placement.partition} "
        f"qos={placement.qos or 'none'} cpus={placement.request_cpus} reason={placement.reason}"
    )
    return True


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    validate_input(root_dir, args)
    state_file = build_state_file(args)
    stages = load_state(state_file, args)
    if stages is None:
        stages = {
            "qwen": StageState(
                name="qwen_collect",
                kind="collect",
                judge_name="qwen397",
                base_url=args.qwen_base_url,
                sbatch_path=str(root_dir / "slurm" / "sharanga" / "local_committee" / "benchmark_collect_eval.sbatch"),
                cpus=4,
            ),
            "mistral": StageState(
                name="mistral_collect",
                kind="collect",
                judge_name="mistral4",
                base_url=args.mistral_base_url,
                sbatch_path=str(root_dir / "slurm" / "sharanga" / "local_committee" / "benchmark_collect_eval.sbatch"),
                cpus=4,
            ),
            "deepseek": StageState(
                name="deepseek_collect",
                kind="collect",
                judge_name="deepseek32",
                base_url=args.deepseek_base_url,
                sbatch_path=str(root_dir / "slurm" / "sharanga" / "local_committee" / "benchmark_collect_eval.sbatch"),
                cpus=4,
            ),
            "merge": StageState(
                name="final_merge",
                kind="merge",
                judge_name="",
                base_url="",
                sbatch_path=str(root_dir / "slurm" / "sharanga" / "local_committee" / "benchmark_final_merge.sbatch"),
                cpus=4,
            ),
        }
    else:
        print_status(f"resuming_watch_state={state_file}")
    print_status(f"watch_state={state_file}")
    print_status("policy=compute_only(no_fallback)")

    while True:
        for stage in stages.values():
            refresh_stage(stage, args)

        all_collect_done = all(stage_done(stages[key]) for key in ("qwen", "mistral", "deepseek"))
        merge_done = stage_done(stages["merge"])

        for key in ("qwen", "mistral", "deepseek"):
            maybe_submit_collect(stages[key], args, root_dir)

        if all_collect_done and not merge_done:
            maybe_submit_merge(stages["merge"], args, root_dir)

        save_state(state_file, stages, args)

        if all_collect_done and merge_done:
            print_status("pipeline_complete=1")
            return 0

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())

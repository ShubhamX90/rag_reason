#!/usr/bin/env python3
# run_evaluation.py
"""
CATS v2.0 - Main Evaluation Script
==================================
Run RAG evaluation with multi-judge committee.

Usage:
    python run_evaluation.py --input data/input.jsonl --config configs/default.yaml
    python run_evaluation.py --input data/input.jsonl --committee conservative
"""

import argparse
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_eval import (
    EvaluationConfig,
    EnhancedEvaluator,
    load_dataset,
    logger,
    setup_file_logging,
    create_default_committee,
)
from rag_eval.config import (
    EnhancedConflictEvalConfig,
    create_cli_committee,
    create_mixed_committee,
    get_claude_cli_judge,
)


def _load_yaml_config(path: str) -> dict:
    """Load YAML config file. Returns {} if path is None or file unreadable."""
    if not path:
        return {}
    try:
        import yaml  # PyYAML
    except ImportError:
        logger.warning("PyYAML not installed; --config will be ignored. Run: pip install pyyaml")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {path}")
        return data
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        return {}
    except Exception as e:
        logger.error(f"Failed to parse YAML config {path}: {e}")
        return {}


def _apply_yaml_to_config(yaml_data: dict, config: EvaluationConfig, args) -> EvaluationConfig:
    """Overlay parsed YAML onto a base EvaluationConfig.

    Honors: outputs_dir, report_md, detailed_results_json,
    pipeline.*, conflict_eval.* (committee voting_strategy / max_concurrent_requests
    / priority_overrides, correct_refusal_full_credit, require_cross_doc_verification,
    max_claims_per_answer).
    """
    if not yaml_data:
        return config

    # Top-level paths
    for k in ("outputs_dir", "report_md", "detailed_results_json"):
        if k in yaml_data:
            setattr(config, k, yaml_data[k])

    # Pipeline section
    pipeline_section = yaml_data.get("pipeline") or {}
    for k in ("batch_size", "verbose", "show_progress", "skip_on_error"):
        if k in pipeline_section:
            setattr(config.pipeline, k, pipeline_section[k])

    # Conflict-eval section
    ce = yaml_data.get("conflict_eval") or {}
    for k in ("correct_refusal_full_credit", "require_cross_doc_verification",
              "max_claims_per_answer", "allow_paraphrases", "aggregate_by_conflict_type",
              "min_entail_confidence", "majority_support_rule", "neutral_as_support",
              "partial_credit_fg"):
        if k in ce:
            setattr(config.conflict, k, ce[k])

    # ignore_contradictions_types is a list in YAML but a tuple in the dataclass.
    if "ignore_contradictions_types" in ce:
        setattr(config.conflict, "ignore_contradictions_types", tuple(ce["ignore_contradictions_types"]))

    # Committee section: voting strategy + priority overrides + CLI type
    committee_section = ce.get("committee") or {}

    committee_type = committee_section.get("type")
    if committee_type == "cli":
        # Rebuild as CLI committee regardless of --committee arg.
        use_codex = bool(committee_section.get("use_codex", True))
        use_claude = bool(committee_section.get("use_claude", True))
        cli_model = str(committee_section.get("cli_model", "sonnet"))
        codex_model = committee_section.get("codex_model")  # None → codex CLI default
        max_conc = int(committee_section.get("max_concurrent_requests", 4))
        config.conflict.use_judge_committee = True
        config.conflict.committee = create_cli_committee(
            use_codex=use_codex,
            use_claude=use_claude,
            cli_model=cli_model,
            codex_model=codex_model,
            max_concurrent_requests=max_conc,
        )
        config.conflict.nli_judge = get_claude_cli_judge(
            model_alias=cli_model,
            model_id="claude-cli/sonnet-nli",
            priority=1,
        )
        logger.info(f"YAML override: CLI committee (use_claude={use_claude}, use_codex={use_codex}, model={cli_model})")
    elif committee_type == "mixed":
        codex_model = committee_section.get("codex_model", "gpt-5.4")
        codex_priority = int(committee_section.get("codex_priority", 4))
        max_conc = int(committee_section.get("max_concurrent_requests", 4))

        # openrouter_judges: list of {model_id, priority} dicts; None → factory default
        raw_or_judges = committee_section.get("openrouter_judges")
        or_judges = None
        if raw_or_judges:
            or_judges = [(j["model_id"], int(j.get("priority", 2))) for j in raw_or_judges]

        config.conflict.use_judge_committee = True
        config.conflict.committee = create_mixed_committee(
            codex_model=codex_model,
            codex_priority=codex_priority,
            openrouter_judges=or_judges,
            max_concurrent_requests=max_conc,
        )
        config.conflict.nli_judge = get_claude_cli_judge(
            model_alias="sonnet",
            model_id="claude-cli/sonnet-nli",
            priority=1,
        )
        logger.info(f"YAML override: MIXED committee (codex gpt-{codex_model} priority={codex_priority} + {len(config.conflict.committee.judges)-1} OpenRouter judges)")
    elif config.conflict.committee is not None:
        if "voting_strategy" in committee_section:
            config.conflict.committee.voting_strategy = committee_section["voting_strategy"]
        if "max_concurrent_requests" in committee_section:
            config.conflict.committee.max_concurrent_requests = int(committee_section["max_concurrent_requests"])

        # Apply priority overrides — rebuild the OpenRouter committee with new priorities.
        overrides = committee_section.get("priority_overrides")
        if overrides:
            from rag_eval.config import create_default_committee
            config.conflict.committee = create_default_committee(
                priority_overrides=overrides,
                max_concurrent_requests=config.conflict.committee.max_concurrent_requests,
            )

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="CATS v2.0 - Enhanced RAG Evaluation Pipeline"
    )
    
    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",  # Support multiple files
        required=True,
        help="Path(s) to input JSONL file(s) with evaluation data. Can specify multiple files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files (default: outputs). For multiple inputs, subdirectories will be created."
    )
    
    # Committee selection
    parser.add_argument(
        "--committee",
        type=str,
        choices=["default", "cli", "none"],
        default="default",
        help="Judge committee preset: default (OpenRouter API), cli (claude+codex CLI, no API key), none (skip committee)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (optional)"
    )
    
    # Execution options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for parallel evaluation (default: 50)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--process-sequentially",
        action="store_true",
        help="Process multiple files sequentially instead of in parallel (default: parallel)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_config(args, input_file: str, output_dir: str) -> EvaluationConfig:
    """Setup evaluation configuration for a specific input file."""
    
    # Create base config
    config = EvaluationConfig()
    
    # Set paths
    config.input_jsonl = input_file
    config.outputs_dir = output_dir
    config.report_md = f"{output_dir}/eval_report.md"
    config.detailed_results_json = f"{output_dir}/detailed_results.json"
    
    # Set pipeline options
    config.pipeline.batch_size = args.batch_size
    config.pipeline.verbose = args.verbose
    
    # Setup judge committee
    if args.committee == "default":
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            logger.error("OPENROUTER_API_KEY not found in environment!")
            logger.error("Please set it in your .env file or export it:")
            logger.error("  export OPENROUTER_API_KEY=your-key-here")
            logger.error("Or use CLI judges with: --committee cli")
            sys.exit(1)
        config.conflict.use_judge_committee = True
        config.conflict.committee = create_default_committee()
        logger.info("Using default judge committee (Haiku + GPT-5.4 + DeepSeek V3.2 via OpenRouter)")
    elif args.committee == "cli":
        config.conflict.use_judge_committee = True
        config.conflict.committee = create_cli_committee(use_codex=True)
        config.conflict.nli_judge = get_claude_cli_judge(
            model_alias="sonnet",
            model_id="claude-cli/sonnet-nli",
            priority=1,
        )
        logger.info("Using CLI judge committee (claude --print + codex exec, no API key required)")
    else:
        config.conflict.use_judge_committee = False
        logger.info("Using single judge mode")

    # Apply YAML config overrides (if --config was passed). YAML wins over CLI
    # for fields it specifies; CLI args remain authoritative for everything else.
    yaml_data = _load_yaml_config(args.config)
    config = _apply_yaml_to_config(yaml_data, config, args)

    return config


async def process_single_file(args, input_file: str, file_idx: int, total: int):
    """Process a single input file."""
    from datetime import datetime
    
    # Create output directory for this file if processing multiple
    if total > 1:
        file_stem = Path(input_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output_dir}/{file_stem}_{timestamp}"
    else:
        output_dir = args.output_dir
    
    logger.info("=" * 60)
    if total > 1:
        logger.info(f"Processing file [{file_idx + 1}/{total}]: {input_file}")
    else:
        logger.info("CATS v2.0 - Conflict-Aware Trust Score Evaluation")
    logger.info("=" * 60)
    
    # Check input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return None
    
    # Setup configuration
    config = setup_config(args, input_file, output_dir)
    
    # Load dataset
    logger.info(f"Loading dataset from {input_file}...")
    dataset = load_dataset(input_file)
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        logger.info(f"Limited to first {args.max_samples} samples")
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Create evaluator
    evaluator = EnhancedEvaluator(config)
    
    # Run evaluation
    try:
        logger.info("Starting evaluation...")
        results = await evaluator.evaluate_async(dataset)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE: {}".format(input_file))
        logger.info("=" * 60)
        
        if "conflict_overall" in results:
            overall = results["conflict_overall"]
            n = overall["n"]
            n_cr = overall.get("correct_refusals", 0)
            logger.info("\nMetrics Summary:")
            logger.info(f"  Samples evaluated: {n}")
            if n_cr:
                logger.info(f"  Correct refusals: {n_cr} (GR=1.0; excluded from sub-metric averages)")
            logger.info(f"  GR Accuracy:        {overall['gr_accuracy']:.3f}  (n={n})")
            if "gr_f1" in overall:
                logger.info(f"  GR F1 (CATS input): {overall['gr_f1']:.3f}")
            logger.info(f"  Behavior Adherence: {overall['behavior']:.3f}  (n={overall.get('behavior_n', n)})")
            logger.info(f"  Factual Grounding:  {overall['factual_grounding']:.3f}  (n={overall.get('factual_grounding_n', n)})")
            logger.info(f"  Single-Truth Recall:{overall['single_truth_recall']:.3f}  (n={overall.get('single_truth_recall_n', 0)})")

            # Dataset-level F1 (proper precision/recall over TP/FP/FN)
            if "gr_dataset_metrics" in results and results["gr_dataset_metrics"]:
                g = results["gr_dataset_metrics"]
                logger.info(f"  GR Dataset F1: {g['f1']:.3f} "
                            f"(P={g['precision']:.3f}, R={g['recall']:.3f})")

            cats_score = overall.get("cats_score", 0.0)
            logger.info(f"\nCATS Score: {cats_score:.3f}")
        
        if "cost_summary" in results:
            cost = results["cost_summary"]
            logger.info(f"\n{'-'*60}")
            logger.info("Cost Summary")
            logger.info(f"{'-'*60}")
            logger.info(f"Total Cost: ${cost['total_cost_usd']:.4f}")
            logger.info(f"Decisions Made: {cost['decisions_made']}")
            logger.info(f"Average Cost per Decision: ${cost['avg_cost_per_decision']:.6f}")
            
            # Per-model costs (committee judges + NLI judge)
            if "per_judge_costs" in cost:
                logger.info(f"\nPer-Model Costs:")
                for model_id, model_cost in cost["per_judge_costs"].items():
                    display_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    logger.info(f"  {display_name} (committee):")
                    logger.info(f"    Total: ${model_cost['total_cost']:.4f}")
                    logger.info(f"    Requests: {model_cost['requests']}")
                    logger.info(f"    Avg/Request: ${model_cost['avg_cost']:.6f}")
            if "nli_judge_cost" in cost:
                n = cost["nli_judge_cost"]
                display_name = n["model_id"].split("/")[-1] if "/" in n["model_id"] else n["model_id"]
                logger.info(f"  {display_name} (NLI judge):")
                logger.info(f"    Total: ${n['total_cost']:.4f}")
                logger.info(f"    Requests: {n['requests']}")
                logger.info(f"    Avg/Request: ${n['avg_cost']:.6f}")
        
        logger.info(f"\nOutput Files:")
        logger.info(f"  Report: {config.report_md}")
        logger.info(f"  Details: {config.detailed_results_json}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed for {input_file}: {e}", exc_info=True)
        return None
    finally:
        # Cleanup
        await evaluator.close()
        logger.info("Evaluation pipeline closed")


async def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_file_logging("logs")
    
    # Handle single or multiple input files
    input_files = args.input if isinstance(args.input, list) else [args.input]
    
    logger.info("=" * 60)
    logger.info("CATS v2.0 - Conflict-Aware Trust Score Evaluation")
    if len(input_files) > 1:
        logger.info(f"Batch Mode: Processing {len(input_files)} files")
    logger.info("=" * 60)
    
    try:
        if len(input_files) == 1:
            # Single file processing
            await process_single_file(args, input_files[0], 0, 1)
        elif args.process_sequentially:
            # Sequential processing
            logger.info("Processing files sequentially...")
            for idx, input_file in enumerate(input_files):
                await process_single_file(args, input_file, idx, len(input_files))
        else:
            # Parallel processing
            logger.info(f"Processing {len(input_files)} files in parallel...")
            tasks = [
                process_single_file(args, input_file, idx, len(input_files))
                for idx, input_file in enumerate(input_files)
            ]
            await asyncio.gather(*tasks)
            
            logger.info("\n" + "=" * 60)
            logger.info("ALL FILES PROCESSED")
            logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

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
    create_conservative_committee,
)
from rag_eval.config import EnhancedConflictEvalConfig


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
        choices=["default", "conservative", "none"],
        default="default",
        help="Judge committee preset: default (Haiku+DeepSeek+Qwen), conservative (cheaper), none (single judge)"
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
    
    # Validate API keys before setting up committee
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if not anthropic_key:
        logger.error("ANTHROPIC_API_KEY not found in environment!")
        logger.error("Please set it in your .env file or export it:")
        logger.error("  export ANTHROPIC_API_KEY=sk-ant-your-key-here")
        sys.exit(1)
    
    if not openrouter_key and args.committee in ["default", "conservative"]:
        logger.error("OPENROUTER_API_KEY not found in environment!")
        logger.error("Please set it in your .env file or export it:")
        logger.error("  export OPENROUTER_API_KEY=your-key-here")
        sys.exit(1)
    
    # Setup judge committee
    if args.committee == "default":
        config.conflict.use_judge_committee = True
        config.conflict.committee = create_default_committee()
        logger.info("Using default judge committee (Haiku + DeepSeek + Qwen + Mistral)")
    elif args.committee == "conservative":
        config.conflict.use_judge_committee = True
        config.conflict.committee = create_conservative_committee()
        logger.info("Using conservative judge committee (lower cost)")
    else:
        config.conflict.use_judge_committee = False
        logger.info("Using single judge mode")
    
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
            logger.info("\nMetrics Summary:")
            logger.info(f"  Samples evaluated: {overall['n']}")
            logger.info(f"  F1_GR: {overall['f1_gr']:.3f}")
            logger.info(f"  Behavior Adherence: {overall['behavior']:.3f}")
            logger.info(f"  Factual Grounding: {overall['factual_grounding']:.3f}")
            logger.info(f"  Single-Truth Recall: {overall['single_truth_recall']:.3f}")
            
            # CATS Score
            import numpy as np
            cats_score = np.mean([
                overall['f1_gr'],
                overall['behavior'],
                overall['factual_grounding'],
                overall['single_truth_recall']
            ])
            logger.info(f"\nCATS Score: {cats_score:.3f}")
        
        if "cost_summary" in results:
            cost = results["cost_summary"]
            logger.info(f"\n{'-'*60}")
            logger.info("Cost Summary")
            logger.info(f"{'-'*60}")
            logger.info(f"Total Cost: ${cost['total_cost_usd']:.4f}")
            logger.info(f"Decisions Made: {cost['decisions_made']}")
            logger.info(f"Average Cost per Decision: ${cost['avg_cost_per_decision']:.6f}")
            
            # Per-model costs
            if "per_judge_costs" in cost:
                logger.info(f"\nPer-Model Costs:")
                for model_id, model_cost in cost["per_judge_costs"].items():
                    # Shorten model names for display
                    display_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    logger.info(f"  {display_name}:")
                    logger.info(f"    Total: ${model_cost['total_cost']:.4f}")
                    logger.info(f"    Requests: {model_cost['requests']}")
                    logger.info(f"    Avg/Request: ${model_cost['avg_cost']:.6f}")
        
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

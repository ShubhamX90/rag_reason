#!/usr/bin/env python3
# run_evaluation_batch.py
"""
CATS v2.0 - Batch Evaluation Script
====================================
Run RAG evaluation on multiple input files in parallel from a single terminal.

Usage:
    # Process multiple files in parallel
    python run_evaluation_batch.py --inputs file1.jsonl file2.jsonl file3.jsonl --committee default
    
    # Use glob patterns
    python run_evaluation_batch.py --inputs data/*.jsonl --committee default
    
    # Specify output directory prefix
    python run_evaluation_batch.py --inputs data/*.jsonl --output-prefix outputs
"""

import argparse
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import List
from datetime import datetime

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="CATS v2.0 - Batch RAG Evaluation Pipeline (Process Multiple Files in Parallel)"
    )
    
    # Input files
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input JSONL files (can specify multiple files or use wildcards)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs",
        help="Directory prefix for output files (default: outputs). Each input will get a separate subdirectory."
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
        help="Batch size for parallel evaluation within each file (default: 50)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples to evaluate per file (for testing)"
    )
    parser.add_argument(
        "--max-concurrent-files",
        type=int,
        default=5,
        help="Maximum number of files to process concurrently (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_config_for_file(args, input_file: str, output_dir: str) -> EvaluationConfig:
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
    elif args.committee == "conservative":
        config.conflict.use_judge_committee = True
        config.conflict.committee = create_conservative_committee()
    else:
        config.conflict.use_judge_committee = False
    
    return config


async def evaluate_single_file(
    args,
    input_file: str,
    file_index: int,
    total_files: int
) -> dict:
    """
    Evaluate a single input file.
    Returns a dict with results and metadata.
    """
    # Create output directory for this file
    file_stem = Path(input_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_prefix}/{file_stem}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing file [{file_index+1}/{total_files}]: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*80}\n")
    
    try:
        # Check input file exists
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return {
                "input_file": input_file,
                "success": False,
                "error": "File not found",
                "output_dir": output_dir
            }
        
        # Setup configuration
        config = setup_config_for_file(args, input_file, output_dir)
        
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
        logger.info(f"Starting evaluation for {input_file}...")
        results = await evaluator.evaluate_async(dataset)
        
        # Cleanup
        await evaluator.close()
        
        # Print summary for this file
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETED: {input_file}")
        logger.info(f"{'='*80}")
        
        if "conflict_overall" in results:
            overall = results["conflict_overall"]
            logger.info(f"Samples evaluated: {overall['n']}")
            logger.info(f"F1_GR: {overall['f1_gr']:.3f}")
            logger.info(f"Behavior Adherence: {overall['behavior']:.3f}")
            logger.info(f"Factual Grounding: {overall['factual_grounding']:.3f}")
            logger.info(f"Single-Truth Recall: {overall['single_truth_recall']:.3f}")
            
            # CATS Score
            import numpy as np
            cats_score = np.mean([
                overall['f1_gr'],
                overall['behavior'],
                overall['factual_grounding'],
                overall['single_truth_recall']
            ])
            logger.info(f"CATS Score: {cats_score:.3f}")
        
        if "cost_summary" in results:
            cost = results["cost_summary"]
            logger.info(f"\n{'-'*60}")
            logger.info("Cost Summary")
            logger.info(f"{'-'*60}")
            logger.info(f"Total Cost: ${cost['total_cost_usd']:.4f}")
            
            # Per-model costs
            if "per_judge_costs" in cost:
                logger.info(f"\nPer-Model Costs:")
                for model_id, model_cost in cost["per_judge_costs"].items():
                    display_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    logger.info(f"  {display_name}: ${model_cost['total_cost']:.4f} ({model_cost['requests']} requests)")
        
        logger.info(f"\nOutput Files:")
        logger.info(f"  Report: {config.report_md}")
        logger.info(f"  Details: {config.detailed_results_json}\n")
        
        return {
            "input_file": input_file,
            "success": True,
            "results": results,
            "output_dir": output_dir,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed for {input_file}: {e}", exc_info=True)
        return {
            "input_file": input_file,
            "success": False,
            "error": str(e),
            "output_dir": output_dir
        }


async def main():
    """Main batch evaluation pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_file_logging("logs")
    
    logger.info("="*80)
    logger.info("CATS v2.0 - Batch Evaluation Mode")
    logger.info("Processing Multiple Files in Parallel")
    logger.info("="*80)
    
    # Expand wildcards and validate input files
    input_files: List[str] = []
    for pattern in args.inputs:
        from glob import glob
        matches = glob(pattern)
        if matches:
            input_files.extend(matches)
        else:
            # Try as literal filename
            if Path(pattern).exists():
                input_files.append(pattern)
            else:
                logger.warning(f"No files found matching pattern: {pattern}")
    
    # Remove duplicates
    input_files = list(set(input_files))
    
    if not input_files:
        logger.error("No input files found!")
        sys.exit(1)
    
    logger.info(f"\nFound {len(input_files)} input file(s) to process:")
    for i, f in enumerate(input_files, 1):
        logger.info(f"  {i}. {f}")
    logger.info("")
    
    # Process files with controlled concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent_files)
    
    async def process_with_semaphore(input_file: str, index: int):
        async with semaphore:
            return await evaluate_single_file(args, input_file, index, len(input_files))
    
    # Create tasks for all files
    tasks = [
        process_with_semaphore(input_file, i)
        for i, input_file in enumerate(input_files)
    ]
    
    # Execute all tasks
    logger.info(f"Starting parallel evaluation of {len(input_files)} files...")
    logger.info(f"Max concurrent files: {args.max_concurrent_files}\n")
    
    try:
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("BATCH EVALUATION COMPLETE")
        logger.info("="*80 + "\n")
        
        successful = [r for r in all_results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in all_results if isinstance(r, dict) and not r.get("success")]
        exceptions = [r for r in all_results if isinstance(r, Exception)]
        
        logger.info(f"Total files processed: {len(input_files)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed) + len(exceptions)}")
        
        if successful:
            logger.info("\n✅ Successfully processed:")
            total_cost = 0.0
            for result in successful:
                logger.info(f"  • {result['input_file']}")
                logger.info(f"    Output: {result['output_dir']}")
                if "results" in result and "cost_summary" in result["results"]:
                    cost = result["results"]["cost_summary"]["total_cost_usd"]
                    total_cost += cost
                    logger.info(f"    Cost: ${cost:.4f}")
            
            logger.info(f"\nTotal cost across all files: ${total_cost:.4f}")
        
        if failed:
            logger.info("\n❌ Failed:")
            for result in failed:
                logger.info(f"  • {result['input_file']}: {result.get('error', 'Unknown error')}")
        
        if exceptions:
            logger.info("\n❌ Exceptions:")
            for exc in exceptions:
                logger.error(f"  • {exc}")
        
    except KeyboardInterrupt:
        logger.warning("\nBatch evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

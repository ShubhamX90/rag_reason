#!/usr/bin/env python3
"""
Extract All Possible Metrics from Main Stagewise Results
---------------------------------------------------------
Extracts every available metric from the evaluation reports.

Usage:
  python extract_all_stagewise_metrics.py outputs/main_stagewise
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def extract_contract_metrics(report_path):
    """Extract all contract compliance metrics."""
    with open(report_path) as f:
        data = json.load(f)
    
    return {
        'total_examples': data.get('total', 0),
        'contract_ok_count': data.get('ok_all_checks', 0),
        'contract_ok_pct': data.get('ok_rate_pct', 0),
        'abstain_count': data.get('abstain_count', 0),
        'problems': data.get('problems', {}),
        'label_f1': data.get('label_f1', {}),
    }


def extract_doc_verdict_metrics(report_path):
    """Extract all document verdict metrics."""
    with open(report_path) as f:
        data = json.load(f)
    
    metrics = {}
    
    # Totals
    totals = data.get('totals', {})
    metrics['total_docs_evaluated'] = totals.get('total_docs', 0)
    metrics['doc_verdict_micro_acc'] = totals.get('micro_accuracy_doc_level', 0)
    metrics['doc_verdict_micro_precision'] = totals.get('micro_precision', 0)
    metrics['doc_verdict_micro_recall'] = totals.get('micro_recall', 0)
    metrics['doc_verdict_micro_f1'] = totals.get('micro_f1', 0)
    
    # Overall (macro)
    overall = data.get('overall', {})
    metrics['doc_verdict_macro_precision'] = overall.get('macro_precision', 0)
    metrics['doc_verdict_macro_recall'] = overall.get('macro_recall', 0)
    metrics['doc_verdict_macro_f1'] = overall.get('macro_f1', 0)
    
    # Per-class metrics
    per_class = data.get('per_class', {})
    for verdict_type in ['supports', 'partially_supports', 'irrelevant']:
        if verdict_type in per_class:
            class_data = per_class[verdict_type]
            metrics[f'{verdict_type}_precision'] = class_data.get('precision', 0)
            metrics[f'{verdict_type}_recall'] = class_data.get('recall', 0)
            metrics[f'{verdict_type}_f1'] = class_data.get('f1', 0)
            metrics[f'{verdict_type}_support'] = class_data.get('support', 0)
    
    return metrics


def extract_conflict_metrics(report_path):
    """Extract all conflict type metrics."""
    with open(report_path) as f:
        data = json.load(f)
    
    metrics = {}
    
    # Overall
    overall = data.get('overall', {})
    metrics['conflict_accuracy'] = overall.get('accuracy', 0)
    metrics['conflict_macro_precision'] = overall.get('macro_precision', 0)
    metrics['conflict_macro_recall'] = overall.get('macro_recall', 0)
    metrics['conflict_macro_f1'] = overall.get('macro_f1', 0)
    metrics['conflict_weighted_precision'] = overall.get('weighted_precision', 0)
    metrics['conflict_weighted_recall'] = overall.get('weighted_recall', 0)
    metrics['conflict_weighted_f1'] = overall.get('weighted_f1', 0)
    
    # Per-class metrics
    per_class = data.get('per_class', {})
    for conflict_type, class_data in per_class.items():
        safe_name = conflict_type.replace(' ', '_').replace('-', '_').lower()
        metrics[f'conflict_{safe_name}_precision'] = class_data.get('precision', 0)
        metrics[f'conflict_{safe_name}_recall'] = class_data.get('recall', 0)
        metrics[f'conflict_{safe_name}_f1'] = class_data.get('f1', 0)
        metrics[f'conflict_{safe_name}_support'] = class_data.get('support', 0)
    
    return metrics


def extract_validation_metrics(oracle_dir):
    """Extract validation funnel metrics."""
    metrics = {}
    
    # Call 1
    call1_path = oracle_dir / "call1_outputs.jsonl"
    if call1_path.exists():
        call1_total = call1_valid = 0
        with open(call1_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                call1_total += 1
                if obj.get('valid', False):
                    call1_valid += 1
        
        metrics['call1_total'] = call1_total
        metrics['call1_valid'] = call1_valid
        metrics['call1_valid_pct'] = 100.0 * call1_valid / call1_total if call1_total > 0 else 0
    
    # Call 2
    call2_path = oracle_dir / "call2_outputs.jsonl"
    if call2_path.exists():
        call2_total = call2_valid = 0
        with open(call2_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                call2_total += 1
                if obj.get('valid', False) or obj.get('has_sentinel', False):
                    call2_valid += 1
        
        metrics['call2_total'] = call2_total
        metrics['call2_valid'] = call2_valid
        metrics['call2_valid_pct'] = 100.0 * call2_valid / call2_total if call2_total > 0 else 0
    
    # Call 3
    call3_path = oracle_dir / "call3_outputs.jsonl"
    if call3_path.exists():
        call3_total = call3_valid = 0
        with open(call3_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                call3_total += 1
                if obj.get('has_sentinel', False):
                    call3_valid += 1
        
        metrics['call3_total'] = call3_total
        metrics['call3_valid'] = call3_valid
        metrics['call3_valid_pct'] = 100.0 * call3_valid / call3_total if call3_total > 0 else 0
    
    # Combined
    combined_path = oracle_dir / "combined.raw.jsonl"
    if combined_path.exists():
        with open(combined_path) as f:
            combined_count = sum(1 for line in f if line.strip())
        metrics['combined_examples'] = combined_count
    
    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_all_stagewise_metrics.py <main_stagewise_dir>")
        print("Example: python extract_all_stagewise_metrics.py outputs/main_stagewise")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)
    
    # Find all oracle directories
    oracle_dirs = {}
    for oracle in ['e2e', 'oracle1', 'oracle2', 'oracle3']:
        oracle_path = base_dir / oracle / 'test'
        if oracle_path.exists():
            oracle_dirs[oracle] = oracle_path
    
    if not oracle_dirs:
        print(f"Error: No oracle directories found in {base_dir}")
        sys.exit(1)
    
    # Extract metrics for each oracle
    all_metrics = {}
    
    for oracle_name, oracle_path in sorted(oracle_dirs.items()):
        print(f"\nExtracting metrics for: {oracle_name}")
        
        metrics = {}
        reports_dir = oracle_path / 'reports'
        
        # Validation metrics
        validation = extract_validation_metrics(oracle_path)
        metrics.update(validation)
        
        # Contract metrics
        contract_path = reports_dir / 'contract.json'
        if contract_path.exists():
            contract = extract_contract_metrics(contract_path)
            metrics.update(contract)
        
        # Doc verdict metrics
        doc_path = reports_dir / 'doc_verdicts.json'
        if doc_path.exists():
            doc_verdict = extract_doc_verdict_metrics(doc_path)
            metrics.update(doc_verdict)
        
        # Conflict metrics
        conflict_path = reports_dir / 'conflict_type.json'
        if conflict_path.exists():
            conflict = extract_conflict_metrics(conflict_path)
            metrics.update(conflict)
        
        all_metrics[oracle_name] = metrics
    
    # Print comprehensive table
    print("\n" + "="*100)
    print("COMPREHENSIVE METRICS TABLE")
    print("="*100)
    
    # Validation funnel
    print("\n### VALIDATION FUNNEL ###")
    print(f"{'Oracle':<12} {'Call1':<15} {'Call2':<15} {'Call3':<15} {'Final':<10}")
    print("-" * 80)
    for oracle in ['e2e', 'oracle1', 'oracle2', 'oracle3']:
        if oracle not in all_metrics:
            continue
        m = all_metrics[oracle]
        
        call1 = f"{m.get('call1_valid', 0)}/{m.get('call1_total', 0)} ({m.get('call1_valid_pct', 0):.1f}%)"
        call2 = f"{m.get('call2_valid', 0)}/{m.get('call2_total', 0)} ({m.get('call2_valid_pct', 0):.1f}%)"
        call3 = f"{m.get('call3_valid', 'N/A')}/{m.get('call3_total', 'N/A')}" if 'call3_total' in m else "N/A"
        final = str(m.get('combined_examples', 0))
        
        print(f"{oracle:<12} {call1:<15} {call2:<15} {call3:<15} {final:<10}")
    
    # Contract compliance
    print("\n### CONTRACT COMPLIANCE ###")
    print(f"{'Oracle':<12} {'Total':<8} {'OK':<8} {'OK%':<8} {'Abstain':<10}")
    print("-" * 50)
    for oracle in ['e2e', 'oracle1', 'oracle2', 'oracle3']:
        if oracle not in all_metrics:
            continue
        m = all_metrics[oracle]
        
        total = m.get('total_examples', 0)
        ok = m.get('contract_ok_count', 0)
        ok_pct = m.get('contract_ok_pct', 0)
        abstain = m.get('abstain_count', 0)
        
        print(f"{oracle:<12} {total:<8} {ok:<8} {ok_pct:<8.1f} {abstain:<10}")
    
    # Doc verdicts
    print("\n### DOCUMENT VERDICTS ###")
    print(f"{'Oracle':<12} {'Micro Acc%':<12} {'Macro F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    for oracle in ['e2e', 'oracle1', 'oracle2', 'oracle3']:
        if oracle not in all_metrics:
            continue
        m = all_metrics[oracle]
        
        micro_acc = m.get('doc_verdict_micro_acc', 0)
        macro_f1 = m.get('doc_verdict_macro_f1', 0)
        precision = m.get('doc_verdict_macro_precision', 0)
        recall = m.get('doc_verdict_macro_recall', 0)
        
        print(f"{oracle:<12} {micro_acc:<12.2f} {macro_f1:<12.4f} {precision:<12.4f} {recall:<12.4f}")
    
    # Conflict type
    print("\n### CONFLICT TYPE ###")
    print(f"{'Oracle':<12} {'Accuracy%':<12} {'Macro F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    for oracle in ['e2e', 'oracle1', 'oracle2', 'oracle3']:
        if oracle not in all_metrics:
            continue
        m = all_metrics[oracle]
        
        acc = m.get('conflict_accuracy', 0)
        f1 = m.get('conflict_macro_f1', 0)
        precision = m.get('conflict_macro_precision', 0)
        recall = m.get('conflict_macro_recall', 0)
        
        print(f"{oracle:<12} {acc:<12.2f} {f1:<12.4f} {precision:<12.4f} {recall:<12.4f}")
    
    # Per-class doc verdicts
    print("\n### DOC VERDICTS PER CLASS ###")
    for verdict in ['supports', 'partially_supports', 'irrelevant']:
        print(f"\n{verdict.upper()}:")
        print(f"{'Oracle':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 58)
        for oracle in ['e2e', 'oracle1', 'oracle2', 'oracle3']:
            if oracle not in all_metrics:
                continue
            m = all_metrics[oracle]
            
            prec = m.get(f'{verdict}_precision', 0)
            rec = m.get(f'{verdict}_recall', 0)
            f1 = m.get(f'{verdict}_f1', 0)
            sup = m.get(f'{verdict}_support', 0)
            
            print(f"{oracle:<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {sup:<10}")
    
    # Export to JSON
    output_json = Path(sys.argv[1]).parent / "all_stagewise_metrics.json"
    with open(output_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"✓ Complete metrics saved to: {output_json}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
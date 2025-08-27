#!/usr/bin/env python3
"""
Point Detector Parameter Sweep Evaluation Script

This script runs the astrocap point detector evaluator with different point_threshold values
to find the optimal threshold for your dataset.
"""

import json
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Optional
import argparse


@dataclass
class EvaluationResult:
    threshold: int
    precision: float
    recall: float
    f1_score: float
    total_ground_truth: int
    total_detections: int
    total_true_positives: int
    frames_processed: int
    success: bool
    error: Optional[str] = None


def create_temp_config(base_config_path: Path, threshold: int, output_file: str) -> Path:
    """Create a temporary config file with the specified threshold."""
    with open(base_config_path, 'r') as f:
        config_content = f.read()

    # Replace the point_threshold value
    lines = config_content.split('\n')
    for i, line in enumerate(lines):
        if 'point_threshold = ' in line:
            lines[i] = f'point_threshold = {threshold}'
        elif 'output_path = ' in line and 'astrocap_sink_point_detector_evaluator' in config_content:
            lines[i] = f'output_path = "{output_file}"'

    # Create temporary file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
    temp_config.write('\n'.join(lines))
    temp_config.close()

    return Path(temp_config.name)


def run_evaluation(config_path: Path, threshold: int, verbose: bool = False) -> EvaluationResult:
    """Run the evaluation with the given config and return results."""
    output_file = f"eval_results_threshold_{threshold}.json"
    temp_config = create_temp_config(config_path, threshold, output_file)

    try:
        # Run the astrocap evaluation
        cmd = [
            'cargo', 'run', '--release', '--bin', 'astrocap_main', '--',
            '--config', str(temp_config)
        ]

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            return EvaluationResult(
                threshold=threshold,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                total_ground_truth=0,
                total_detections=0,
                total_true_positives=0,
                frames_processed=0,
                success=False,
                error=f"Process failed: {result.stderr}"
            )

        # Parse the JSON results
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                results = json.load(f)

            metrics = results['aggregate_metrics']
            return EvaluationResult(
                threshold=threshold,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                total_ground_truth=metrics['total_ground_truth'],
                total_detections=metrics['total_detections'],
                total_true_positives=metrics['total_true_positives'],
                frames_processed=metrics['frames_processed'],
                success=True
            )
        else:
            return EvaluationResult(
                threshold=threshold,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                total_ground_truth=0,
                total_detections=0,
                total_true_positives=0,
                frames_processed=0,
                success=False,
                error="Results file not created"
            )

    except subprocess.TimeoutExpired:
        return EvaluationResult(
            threshold=threshold,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            total_ground_truth=0,
            total_detections=0,
            total_true_positives=0,
            frames_processed=0,
            success=False,
            error="Timeout"
        )
    except Exception as e:
        return EvaluationResult(
            threshold=threshold,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            total_ground_truth=0,
            total_detections=0,
            total_true_positives=0,
            frames_processed=0,
            success=False,
            error=str(e)
        )
    finally:
        # Cleanup temp config
        if temp_config.exists():
            temp_config.unlink()


def print_results_table(results: List[EvaluationResult]):
    """Print a nicely formatted results table."""
    print("\n" + "=" * 80)
    print("POINT DETECTOR THRESHOLD EVALUATION RESULTS")
    print("=" * 80)

    # Header
    print(
        f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'GT':<8} {'Det':<8} {'TP':<8} {'Status':<10}")
    print("-" * 80)

    # Results
    for result in results:
        if result.success:
            status = "✓ OK"
            precision_str = f"{result.precision:.3f}"
            recall_str = f"{result.recall:.3f}"
            f1_str = f"{result.f1_score:.3f}"
        else:
            status = "✗ FAIL"
            precision_str = "---"
            recall_str = "---"
            f1_str = "---"

        print(f"{result.threshold:<10} {precision_str:<10} {recall_str:<10} {f1_str:<10} "
              f"{result.total_ground_truth:<8} {result.total_detections:<8} {result.total_true_positives:<8} {status:<10}")

    print("-" * 80)


def print_best_results(results: List[EvaluationResult]):
    """Print the best results by different metrics."""
    successful_results = [r for r in results if r.success]

    if not successful_results:
        print("\n❌ No successful evaluations!")
        return

    # Find best by different metrics
    best_f1 = max(successful_results, key=lambda x: x.f1_score)
    best_precision = max(successful_results, key=lambda x: x.precision)
    best_recall = max(successful_results, key=lambda x: x.recall)

    print("\n" + "=" * 60)
    print("🏆 BEST RESULTS")
    print("=" * 60)

    print(
        f"🎯 Best F1-Score:   threshold={best_f1.threshold:<3} F1={best_f1.f1_score:.3f} (P={best_f1.precision:.3f}, R={best_recall.recall:.3f})")
    print(f"🎯 Best Precision:  threshold={best_precision.threshold:<3} P={best_precision.precision:.3f}")
    print(f"🎯 Best Recall:     threshold={best_recall.threshold:<3} R={best_recall.recall:.3f}")

    # Overall recommendation
    print(f"\n💡 RECOMMENDED: Use threshold = {best_f1.threshold} (best F1-score = {best_f1.f1_score:.3f})")
    print("=" * 60)


def cleanup_results_files(thresholds: List[int]):
    """Clean up temporary result files."""
    for threshold in thresholds:
        result_file = Path(f"eval_results_threshold_{threshold}.json")
        if result_file.exists():
            result_file.unlink()


def main():
    parser = argparse.ArgumentParser(description='Evaluate point detector with different thresholds')
    parser.add_argument('config', help='Base config file (e.g., astrocap_eval_pd_peak.toml)')
    parser.add_argument('--min-threshold', type=int, default=20,
                        help='Minimum threshold to test (default: 20)')
    parser.add_argument('--max-threshold', type=int, default=100,
                        help='Maximum threshold to test (default: 100)')
    parser.add_argument('--step', type=int, default=10,
                        help='Step size between thresholds (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--keep-results', action='store_true',
                        help='Keep individual result JSON files')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    # Generate threshold values
    thresholds = list(range(args.min_threshold, args.max_threshold + 1, args.step))

    print(f"🚀 Starting evaluation with thresholds: {thresholds}")
    print(f"📁 Base config: {config_path}")
    print(f"⏱️  This may take a while...\n")

    results = []

    for i, threshold in enumerate(thresholds, 1):
        print(f"[{i}/{len(thresholds)}] Evaluating threshold {threshold}...", end="", flush=True)

        result = run_evaluation(config_path, threshold, args.verbose)
        results.append(result)

        if result.success:
            print(f" ✓ F1={result.f1_score:.3f}")
        else:
            print(f" ✗ Failed: {result.error}")

    # Print results
    print_results_table(results)
    print_best_results(results)

    # Cleanup
    if not args.keep_results:
        cleanup_results_files(thresholds)
        print(f"\n🧹 Cleaned up temporary result files")
    else:
        print(f"\n📁 Result files kept: eval_results_threshold_*.json")


if __name__ == '__main__':
    main()

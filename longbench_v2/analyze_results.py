"""
Analyze results from LongBench-v2 evaluation runs.

This script provides detailed analysis of RLM evaluation results on LongBench-v2.

Usage:
    python analyze_results.py results/eval_1.json
    python analyze_results.py results/eval_1.json --detailed
    python analyze_results.py results/eval_1.json results/eval_2.json --compare
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary(results: Dict[str, Any], label: str = ""):
    """Print summary statistics."""
    metadata = results.get('metadata', {})
    
    print(f"\n{'='*80}")
    print(f"SUMMARY{f' - {label}' if label else ''}")
    print(f"{'='*80}")
    
    print(f"\nConfiguration:")
    print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"  Model: {metadata.get('model', 'N/A')}")
    print(f"  Recursive Model: {metadata.get('recursive_model', 'N/A')}")
    print(f"  Max Iterations: {metadata.get('max_iterations', 'N/A')}")
    
    print(f"\nFilters:")
    filters = metadata.get('filters', {})
    print(f"  Domain: {filters.get('domain') or 'None'}")
    print(f"  Difficulty: {filters.get('difficulty') or 'None'}")
    print(f"  Length: {filters.get('length') or 'None'}")
    
    print(f"\nResults:")
    print(f"  Total Examples: {metadata.get('num_examples', 0)}")
    print(f"  Successful: {metadata.get('successful', 0)}")
    print(f"  Correct: {metadata.get('correct', 0)}")
    print(f"  Accuracy: {metadata.get('accuracy', 0):.2f}%")


def print_detailed_analysis(results: Dict[str, Any]):
    """Print detailed breakdown of results."""
    result_list = results.get('results', [])
    
    if not result_list:
        print("\nNo results to analyze.")
        return
    
    # Filter successful results
    successful = [r for r in result_list if r.get('success', False)]
    
    if not successful:
        print("\nNo successful results to analyze.")
        return
    
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    # By domain
    domain_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'context_lengths': []})
    for r in successful:
        domain = r.get('domain', 'unknown')
        domain_stats[domain]['total'] += 1
        if r.get('is_correct', False):
            domain_stats[domain]['correct'] += 1
        if 'context_length' in r:
            domain_stats[domain]['context_lengths'].append(r['context_length'])
    
    print(f"\nBy Domain:")
    for domain, stats in sorted(domain_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_len = sum(stats['context_lengths']) / len(stats['context_lengths']) if stats['context_lengths'] else 0
        print(f"  {domain}:")
        print(f"    Accuracy: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        print(f"    Avg Context Length: {avg_len:,.0f} chars")
    
    # By difficulty
    difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in successful:
        difficulty = r.get('difficulty', 'unknown')
        difficulty_stats[difficulty]['total'] += 1
        if r.get('is_correct', False):
            difficulty_stats[difficulty]['correct'] += 1
    
    print(f"\nBy Difficulty:")
    for difficulty, stats in sorted(difficulty_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {difficulty}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # By length
    length_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in successful:
        length = r.get('length', 'unknown')
        length_stats[length]['total'] += 1
        if r.get('is_correct', False):
            length_stats[length]['correct'] += 1
    
    print(f"\nBy Length Category:")
    for length, stats in sorted(length_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {length}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Context length distribution
    context_lengths = [r.get('context_length', 0) for r in successful if 'context_length' in r]
    if context_lengths:
        print(f"\nContext Length Statistics:")
        print(f"  Min: {min(context_lengths):,} chars")
        print(f"  Max: {max(context_lengths):,} chars")
        print(f"  Mean: {sum(context_lengths) / len(context_lengths):,.0f} chars")
        print(f"  Median: {sorted(context_lengths)[len(context_lengths)//2]:,} chars")
    
    # Error analysis
    failed = [r for r in result_list if not r.get('success', False)]
    if failed:
        print(f"\nFailed Examples: {len(failed)}")
        error_types = defaultdict(int)
        for r in failed:
            error = r.get('error', 'unknown')
            # Simplify error message
            if 'timeout' in error.lower():
                error_types['Timeout'] += 1
            elif 'rate limit' in error.lower():
                error_types['Rate Limit'] += 1
            else:
                error_types['Other'] += 1
        
        print("  Error types:")
        for error_type, count in sorted(error_types.items()):
            print(f"    {error_type}: {count}")


def print_comparison(results_list: List[tuple]):
    """Compare multiple result files."""
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    # Print side-by-side comparison
    print(f"\n{'Metric':<30}", end="")
    for label, _ in results_list:
        print(f"{label:<20}", end="")
    print()
    
    print("-" * 80)
    
    # Accuracy
    print(f"{'Accuracy':<30}", end="")
    for _, results in results_list:
        acc = results.get('metadata', {}).get('accuracy', 0)
        print(f"{acc:>6.2f}%{'':<13}", end="")
    print()
    
    # Model
    print(f"{'Model':<30}", end="")
    for _, results in results_list:
        model = results.get('metadata', {}).get('model', 'N/A')[:18]
        print(f"{model:<20}", end="")
    print()
    
    # Total examples
    print(f"{'Total Examples':<30}", end="")
    for _, results in results_list:
        num = results.get('metadata', {}).get('num_examples', 0)
        print(f"{num:<20}", end="")
    print()
    
    # Successful
    print(f"{'Successful':<30}", end="")
    for _, results in results_list:
        num = results.get('metadata', {}).get('successful', 0)
        print(f"{num:<20}", end="")
    print()
    
    # Correct
    print(f"{'Correct':<30}", end="")
    for _, results in results_list:
        num = results.get('metadata', {}).get('correct', 0)
        print(f"{num:<20}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze LongBench-v2 evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic summary
  python analyze_results.py results/eval_1.json
  
  # Detailed analysis
  python analyze_results.py results/eval_1.json --detailed
  
  # Compare multiple runs
  python analyze_results.py results/eval_1.json results/eval_2.json --compare
        """
    )
    
    parser.add_argument(
        'result_files',
        nargs='+',
        help='Path(s) to result JSON file(s)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed breakdown by domain, difficulty, and length'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple result files'
    )
    
    args = parser.parse_args()
    
    # Load all result files
    results_list = []
    for file_path in args.result_files:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            continue
        
        results = load_results(file_path)
        label = Path(file_path).stem
        results_list.append((label, results))
    
    if not results_list:
        print("No valid result files loaded.")
        return
    
    # Single file analysis
    if len(results_list) == 1:
        label, results = results_list[0]
        print_summary(results, label)
        
        if args.detailed:
            print_detailed_analysis(results)
    
    # Multiple file comparison
    elif len(results_list) > 1:
        if args.compare:
            print_comparison(results_list)
            print()
        
        # Also show individual summaries
        for label, results in results_list:
            print_summary(results, label)
            if args.detailed:
                print_detailed_analysis(results)
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()


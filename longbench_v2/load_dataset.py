"""
Load and prepare LongBench-v2 dataset for RLM evaluation.

This script downloads the LongBench-v2 dataset from Hugging Face and saves it locally.

Usage:
    python load_dataset.py --output datasets/longbench_v2.json
    python load_dataset.py --output datasets/longbench_v2.json --filter-domain "single_document_qa"
    python load_dataset.py --output datasets/longbench_v2.json --filter-difficulty easy
    python load_dataset.py --output datasets/longbench_v2.json --filter-length short
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def load_longbench_v2(
    filter_domain: Optional[str] = None,
    filter_difficulty: Optional[str] = None,
    filter_length: Optional[str] = None,
    max_samples: Optional[int] = None
) -> list:
    """
    Load LongBench-v2 dataset from Hugging Face.
    
    Args:
        filter_domain: Filter by domain (e.g., 'single_document_qa', 'multi_document_qa')
        filter_difficulty: Filter by difficulty ('easy' or 'hard')
        filter_length: Filter by length ('short', 'medium', or 'long')
        max_samples: Maximum number of samples to load
    
    Returns:
        List of dataset entries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found. Install it with:")
        print("  pip install datasets")
        return []
    
    print("Loading LongBench-v2 dataset from Hugging Face...")
    print("This may take a while on first download...")
    
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    # Convert to list of dicts
    data = [dict(item) for item in dataset]
    
    print(f"Loaded {len(data)} samples")
    
    # Apply filters
    if filter_domain:
        data = [item for item in data if item.get('domain') == filter_domain]
        print(f"After domain filter ({filter_domain}): {len(data)} samples")
    
    if filter_difficulty:
        data = [item for item in data if item.get('difficulty') == filter_difficulty]
        print(f"After difficulty filter ({filter_difficulty}): {len(data)} samples")
    
    if filter_length:
        data = [item for item in data if item.get('length') == filter_length]
        print(f"After length filter ({filter_length}): {len(data)} samples")
    
    if max_samples and len(data) > max_samples:
        import random
        data = random.sample(data, max_samples)
        print(f"Randomly sampled {max_samples} samples")
    
    return data


def save_dataset(data: list, output_path: str):
    """Save dataset to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to: {output_file}")
    print(f"Total samples: {len(data)}")


def print_dataset_stats(data: list):
    """Print statistics about the dataset."""
    if not data:
        print("No data to analyze")
        return
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    # Count by domain
    domains = {}
    for item in data:
        domain = item.get('domain', 'unknown')
        domains[domain] = domains.get(domain, 0) + 1
    
    print("\nBy Domain:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")
    
    # Count by difficulty
    difficulties = {}
    for item in data:
        difficulty = item.get('difficulty', 'unknown')
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
    
    print("\nBy Difficulty:")
    for difficulty, count in sorted(difficulties.items()):
        print(f"  {difficulty}: {count}")
    
    # Count by length
    lengths = {}
    for item in data:
        length = item.get('length', 'unknown')
        lengths[length] = lengths.get(length, 0) + 1
    
    print("\nBy Length:")
    for length, count in sorted(lengths.items()):
        print(f"  {length}: {count}")
    
    # Sample entry
    if data:
        print("\nSample Entry:")
        sample = data[0]
        print(f"  ID: {sample.get('_id')}")
        print(f"  Domain: {sample.get('domain')}")
        print(f"  Sub-domain: {sample.get('sub_domain')}")
        print(f"  Difficulty: {sample.get('difficulty')}")
        print(f"  Length: {sample.get('length')}")
        print(f"  Question: {sample.get('question', '')[:100]}...")
        print(f"  Context length: {len(sample.get('context', ''))} characters")
        print(f"  Answer: {sample.get('answer')}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Load and prepare LongBench-v2 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download full dataset
  python load_dataset.py --output datasets/longbench_v2.json
  
  # Download only single-document QA examples
  python load_dataset.py --output datasets/single_doc_qa.json --filter-domain single_document_qa
  
  # Download only easy examples
  python load_dataset.py --output datasets/easy.json --filter-difficulty easy
  
  # Download only short examples
  python load_dataset.py --output datasets/short.json --filter-length short
  
  # Download 50 samples for testing
  python load_dataset.py --output datasets/test_sample.json --max-samples 50
  
  # Combine filters
  python load_dataset.py --output datasets/easy_short.json --filter-difficulty easy --filter-length short
        """
    )
    
    parser.add_argument(
        '--output',
        default='datasets/longbench_v2.json',
        help='Output JSON file path (default: datasets/longbench_v2.json)'
    )
    
    parser.add_argument(
        '--filter-domain',
        help='Filter by domain (e.g., single_document_qa, multi_document_qa, long_in_context_learning)'
    )
    
    parser.add_argument(
        '--filter-difficulty',
        choices=['easy', 'hard'],
        help='Filter by difficulty level'
    )
    
    parser.add_argument(
        '--filter-length',
        choices=['short', 'medium', 'long'],
        help='Filter by context length category'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to load (random sample)'
    )
    
    parser.add_argument(
        '--show-stats',
        action='store_true',
        default=True,
        help='Show dataset statistics (default: True)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    data = load_longbench_v2(
        filter_domain=args.filter_domain,
        filter_difficulty=args.filter_difficulty,
        filter_length=args.filter_length,
        max_samples=args.max_samples
    )
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Show statistics
    if args.show_stats:
        print_dataset_stats(data)
    
    # Save dataset
    save_dataset(data, args.output)


if __name__ == "__main__":
    main()


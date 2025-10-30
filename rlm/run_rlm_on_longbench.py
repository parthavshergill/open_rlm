"""
Run RLM evaluation on LongBench-v2 dataset.

This script evaluates the RLM on the LongBench-v2 benchmark, which tests
deep understanding and reasoning on long contexts.

Usage:
    # Run on a single example
    python run_rlm_on_longbench.py --mode single --dataset datasets/longbench_v2.json
    
    # Run on multiple examples
    python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --num-examples 10
    
    # Use specific model
    python run_rlm_on_longbench.py --mode single --model gpt-4o --dataset datasets/longbench_v2.json
    
    # Use Gemini models
    python run_rlm_on_longbench.py --mode single --model gemini-1.5-pro --dataset datasets/longbench_v2.json
    
    # Filter by domain or difficulty
    python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --filter-domain single_document_qa
    python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --filter-difficulty easy
    
    # Save results
    python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --output-results results/run_1.json
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from rlm.rlm_repl import RLM_REPL


def load_dataset(file_path: str) -> list:
    """Load LongBench-v2 dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_dataset(
    data: list,
    filter_domain: Optional[str] = None,
    filter_difficulty: Optional[str] = None,
    filter_length: Optional[str] = None
) -> list:
    """Filter dataset by various criteria."""
    filtered = data
    
    if filter_domain:
        filtered = [item for item in filtered if item.get('domain') == filter_domain]
        print(f"Filtered by domain '{filter_domain}': {len(filtered)} samples")
    
    if filter_difficulty:
        filtered = [item for item in filtered if item.get('difficulty') == filter_difficulty]
        print(f"Filtered by difficulty '{filter_difficulty}': {len(filtered)} samples")
    
    if filter_length:
        filtered = [item for item in filtered if item.get('length') == filter_length]
        print(f"Filtered by length '{filter_length}': {len(filtered)} samples")
    
    return filtered


def format_question(item: Dict[str, Any]) -> str:
    """
    Format the question with multiple choice options.
    
    Args:
        item: Dataset item with question and choices
    
    Returns:
        Formatted question string
    """
    question = item.get('question', '')
    choices = []
    
    for choice_key in ['choice_A', 'choice_B', 'choice_C', 'choice_D']:
        if choice_key in item and item[choice_key]:
            letter = choice_key.split('_')[1]  # Extract A, B, C, or D
            choices.append(f"{letter}. {item[choice_key]}")
    
    formatted = f"{question}\n\n"
    formatted += "\n".join(choices)
    formatted += "\n\nProvide your answer as a single letter (A, B, C, or D)."
    
    return formatted


def extract_answer(response: str) -> str:
    """
    Extract the answer choice (A, B, C, or D) from the model response.
    
    Args:
        response: Raw model response
    
    Returns:
        Extracted answer as single letter (A, B, C, or D) or empty string if not found
    """
    response = response.strip().upper()
    
    # Try to find a clear answer pattern
    # Look for patterns like "A", "Answer: A", "The answer is A", etc.
    for letter in ['A', 'B', 'C', 'D']:
        if response == letter:
            return letter
        if f"ANSWER IS {letter}" in response:
            return letter
        if f"ANSWER: {letter}" in response:
            return letter
        if response.startswith(letter) and (len(response) == 1 or response[1] in ['.', ')', ' ']):
            return letter
    
    # If no clear pattern, look for first occurrence of A, B, C, or D
    for char in response:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    return ""


def evaluate_response(response: str, correct_answer: str) -> bool:
    """
    Evaluate if the response matches the correct answer.
    
    Args:
        response: Model response
        correct_answer: Correct answer (A, B, C, or D)
    
    Returns:
        True if correct, False otherwise
    """
    extracted = extract_answer(response)
    return extracted == correct_answer.upper()


def run_single_example(item: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Run RLM on a single LongBench-v2 example.
    
    Args:
        item: Dataset item
        args: Command line arguments
    
    Returns:
        Result dictionary with prediction and correctness
    """
    # Extract data
    raw_context = item.get('context', '')
    question = format_question(item)
    correct_answer = item.get('answer', '')
    
    # Structure context with metadata for REPL environment
    # This makes it clear to the model that documents are in REPL, not in the prompt
    structured_context = {
        'document': raw_context,
        'metadata': {
            'id': item.get('_id'),
            'domain': item.get('domain'),
            'sub_domain': item.get('sub_domain'),
            'difficulty': item.get('difficulty'),
            'length': item.get('length'),
            'document_length_chars': len(raw_context)
        }
    }
    
    print(f"\n{'='*80}")
    print(f"ID: {item.get('_id')}")
    print(f"Domain: {item.get('domain')} / {item.get('sub_domain')}")
    print(f"Difficulty: {item.get('difficulty')}")
    print(f"Length: {item.get('length')}")
    print(f"Context length: {len(raw_context)} characters")
    print(f"Correct answer: {correct_answer}")
    print(f"{'='*80}")
    
    if args.show_question:
        print(f"\nQuestion:\n{question}\n")
    
    if args.show_context:
        preview_len = 1000
        print(f"\nContext preview (first {preview_len} characters):")
        print(raw_context[:preview_len])
        print("...\n")
    
    # Initialize RLM
    print(f"\n{'-'*80}")
    print("Initializing RLM...")
    print(f"  Model: {args.model}")
    print(f"  Recursive Model: {args.recursive_model}")
    print(f"  Max Iterations: {args.max_iterations}")
    print(f"  Logging: {'Enabled' if args.logging else 'Disabled'}")
    print(f"{'-'*80}\n")
    
    rlm = RLM_REPL(
        model=args.model,
        recursive_model=args.recursive_model,
        enable_logging=args.logging,
        max_iterations=args.max_iterations
    )
    
    # Run RLM with structured context
    # The document is NOT included in the prompt - only loaded into REPL environment
    print("Running RLM...\n")
    print("NOTE: Document is loaded into REPL environment as context['document']")
    print("      Metadata is available at context['metadata']\n")
    try:
        response = rlm.completion(context=structured_context, query=question)
        
        # Evaluate
        is_correct = evaluate_response(response, correct_answer)
        extracted_answer = extract_answer(response)
        
        print(f"\n{'='*80}")
        print("RESULTS:")
        print(f"  Model response: {response}")
        print(f"  Extracted answer: {extracted_answer}")
        print(f"  Correct answer: {correct_answer}")
        print(f"  Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print(f"{'='*80}\n")
        
        return {
            'item_id': item.get('_id'),
            'domain': item.get('domain'),
            'sub_domain': item.get('sub_domain'),
            'difficulty': item.get('difficulty'),
            'length': item.get('length'),
            'question': item.get('question'),
            'context_length': len(raw_context),
            'correct_answer': correct_answer,
            'model_response': response,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'success': True
        }
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*80}\n")
        
        return {
            'item_id': item.get('_id'),
            'domain': item.get('domain'),
            'sub_domain': item.get('sub_domain'),
            'difficulty': item.get('difficulty'),
            'length': item.get('length'),
            'error': str(e),
            'success': False
        }


def run_single_mode(args):
    """Run on a single example from the dataset."""
    print("="*80)
    print("MODE: SINGLE (One example)")
    print("="*80)
    
    # Load dataset
    data = load_dataset(args.dataset)
    
    # Apply filters
    data = filter_dataset(
        data,
        filter_domain=args.filter_domain,
        filter_difficulty=args.filter_difficulty,
        filter_length=args.filter_length
    )
    
    if not data:
        print("No examples found matching filters!")
        return None
    
    # Select random example
    item = random.choice(data)
    
    # Run evaluation
    result = run_single_example(item, args)
    
    return result


def run_batch_mode(args):
    """Run on multiple examples from the dataset."""
    print("="*80)
    print(f"MODE: BATCH ({args.num_examples} examples)")
    print("="*80)
    
    # Load dataset
    data = load_dataset(args.dataset)
    
    # Apply filters
    data = filter_dataset(
        data,
        filter_domain=args.filter_domain,
        filter_difficulty=args.filter_difficulty,
        filter_length=args.filter_length
    )
    
    if not data:
        print("No examples found matching filters!")
        return None
    
    # Select examples
    num_to_test = min(args.num_examples, len(data))
    selected_items = random.sample(data, num_to_test) if not args.sequential else data[:num_to_test]
    
    print(f"\nTesting {num_to_test} examples")
    
    # Run evaluations
    results = []
    for i, item in enumerate(selected_items, 1):
        print(f"\n{'#'*80}")
        print(f"# Example {i}/{num_to_test}")
        print(f"{'#'*80}")
        
        result = run_single_example(item, args)
        results.append(result)
    
    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    correct = [r for r in successful if r.get('is_correct', False)]
    
    accuracy = len(correct) / len(successful) * 100 if successful else 0
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total examples: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Correct: {len(correct)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Breakdown by domain
    domain_stats = {}
    for r in successful:
        domain = r.get('domain', 'unknown')
        if domain not in domain_stats:
            domain_stats[domain] = {'total': 0, 'correct': 0}
        domain_stats[domain]['total'] += 1
        if r.get('is_correct', False):
            domain_stats[domain]['correct'] += 1
    
    if domain_stats:
        print(f"\nBy Domain:")
        for domain, stats in sorted(domain_stats.items()):
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {domain}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    print(f"{'='*80}\n")
    
    # Save results if requested
    if args.output_results:
        save_results(results, args)
    
    return results


def save_results(results: List[Dict[str, Any]], args):
    """Save evaluation results to JSON file."""
    output_file = Path(args.output_results)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    successful = [r for r in results if r.get('success', False)]
    correct = [r for r in successful if r.get('is_correct', False)]
    accuracy = len(correct) / len(successful) * 100 if successful else 0
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset': args.dataset,
            'model': args.model,
            'recursive_model': args.recursive_model,
            'max_iterations': args.max_iterations,
            'num_examples': len(results),
            'successful': len(successful),
            'correct': len(correct),
            'accuracy': accuracy,
            'filters': {
                'domain': args.filter_domain,
                'difficulty': args.filter_difficulty,
                'length': args.filter_length
            }
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run RLM evaluation on LongBench-v2 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single example with default model
  python run_rlm_on_longbench.py --mode single --dataset datasets/longbench_v2.json
  
  # Batch evaluation with 10 examples
  python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --num-examples 10
  
  # Use GPT-4o
  python run_rlm_on_longbench.py --mode single --dataset datasets/longbench_v2.json --model gpt-4o
  
  # Use Gemini
  python run_rlm_on_longbench.py --mode single --dataset datasets/longbench_v2.json --model gemini-1.5-pro
  
  # Filter by domain
  python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --filter-domain single_document_qa --num-examples 5
  
  # Save results
  python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --num-examples 20 --output-results results/eval_1.json
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['single', 'batch'],
        default='single',
        help='single: evaluate one example, batch: evaluate multiple examples'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to LongBench-v2 dataset JSON file'
    )
    
    parser.add_argument(
        '--filter-domain',
        help='Filter by domain (e.g., single_document_qa, multi_document_qa)'
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
    
    # Batch mode options
    parser.add_argument(
        '--num-examples',
        type=int,
        default=10,
        help='Number of examples to test in batch mode (default: 10)'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Process examples sequentially instead of random sampling'
    )
    
    # RLM configuration
    parser.add_argument(
        '--model',
        default='gpt-5-mini-2025-08-07',
        help='Main model to use (default: gpt-5-mini-2025-08-07)'
    )
    
    parser.add_argument(
        '--recursive-model',
        default=None,
        help='Model for recursive calls (default: same as --model)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=20,
        help='Maximum RLM iterations (default: 20)'
    )
    
    parser.add_argument(
        '--no-logging',
        dest='logging',
        action='store_false',
        help='Disable detailed logging'
    )
    
    parser.add_argument(
        '--show-question',
        action='store_true',
        help='Show the full question'
    )
    
    parser.add_argument(
        '--show-context',
        action='store_true',
        help='Show preview of the context'
    )
    
    # Output options
    parser.add_argument(
        '--output-results',
        help='Path to save results JSON file (e.g., results/eval_1.json)'
    )
    
    parser.set_defaults(logging=True)
    
    args = parser.parse_args()
    
    # Set recursive model to main model if not specified
    if args.recursive_model is None:
        args.recursive_model = args.model
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found at {args.dataset}")
        print("\nFirst download the dataset using:")
        print("  python load_dataset.py --output datasets/longbench_v2.json")
        return
    
    # Run selected mode
    if args.mode == 'single':
        run_single_mode(args)
    elif args.mode == 'batch':
        run_batch_mode(args)


if __name__ == "__main__":
    main()


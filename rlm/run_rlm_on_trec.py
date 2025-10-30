"""
Configurable script to test RLM on TREC QC dataset.

Supports OpenAI (GPT) and Google (Gemini) models.

Usage:
    # Demo mode (synthetic data, fast)
    python run_rlm_on_trec.py --mode demo
    
    # Single example from real dataset
    python run_rlm_on_trec.py --mode single
    
    # Multiple examples
    python run_rlm_on_trec.py --mode batch --num-examples 3
    
    # Specific context length
    python run_rlm_on_trec.py --mode single --context-length 32768
    
    # Different OpenAI model (default is gpt-5-mini-2025-08-07)
    python run_rlm_on_trec.py --mode single --model gpt-4o
    
    # Use Gemini models
    python run_rlm_on_trec.py --mode single --model gemini-1.5-pro
    python run_rlm_on_trec.py --mode single --model gemini-1.5-flash
    
    # Mix models (Gemini for main, GPT for recursive)
    python run_rlm_on_trec.py --mode single --model gemini-1.5-pro --recursive-model gpt-5-mini-2025-08-07
    
    # Disable logging for cleaner output
    python run_rlm_on_trec.py --mode single --no-logging
"""

import json
import random
import argparse
from pathlib import Path
from rlm.rlm_repl import RLM_REPL


def create_demo_dataset():
    """Create small synthetic dataset for demo mode."""
    return [
        {
            "text": "What is the capital of France?",
            "label_coarse": "LOC",
            "label_coarse_text": "location",
            "user_id": "U001",
            "date": "01/15/2023"
        },
        {
            "text": "Who invented the telephone?",
            "label_coarse": "HUM",
            "label_coarse_text": "human",
            "user_id": "U002",
            "date": "02/20/2023"
        },
        {
            "text": "When did World War II end?",
            "label_coarse": "NUM",
            "label_coarse_text": "numeric",
            "user_id": "U001",
            "date": "03/10/2023"
        },
        {
            "text": "What does CPU stand for?",
            "label_coarse": "ABBR",
            "label_coarse_text": "abbreviation",
            "user_id": "U003",
            "date": "04/05/2023"
        },
        {
            "text": "How many continents are there?",
            "label_coarse": "NUM",
            "label_coarse_text": "numeric",
            "user_id": "U001",
            "date": "05/12/2023"
        },
        {
            "text": "Where is the Eiffel Tower?",
            "label_coarse": "LOC",
            "label_coarse_text": "location",
            "user_id": "U002",
            "date": "06/18/2023"
        },
        {
            "text": "Who wrote Romeo and Juliet?",
            "label_coarse": "HUM",
            "label_coarse_text": "human",
            "user_id": "U002",
            "date": "07/22/2023"
        },
        {
            "text": "What is the speed of light?",
            "label_coarse": "NUM",
            "label_coarse_text": "numeric",
            "user_id": "U003",
            "date": "08/30/2023"
        },
    ]


def format_dataset_as_string(dataset: list) -> str:
    """
    Convert dataset list to a human-readable string format.
    This forces the model to use llm_query tool to process the data.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("DATASET ENTRIES")
    lines.append("=" * 80)
    lines.append("")
    
    for i, entry in enumerate(dataset, 1):
        lines.append(f"--- Entry {i} ---")
        lines.append(f"Text: {entry.get('text', 'N/A')}")
        lines.append(f"Label (Coarse): {entry.get('label_coarse', 'N/A')}")
        lines.append(f"Label Description: {entry.get('label_coarse_text', 'N/A')}")
        lines.append(f"User ID: {entry.get('user_id', 'N/A')}")
        lines.append(f"Date: {entry.get('date', 'N/A')}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append(f"Total entries: {len(dataset)}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def load_trec_dataset(file_path: str) -> dict:
    """Load the TREC QC dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def run_demo_mode(args):
    """Run on synthetic demo data."""
    print("="*80)
    print("MODE: DEMO (Synthetic Data)")
    print("="*80)
    
    dataset_list = create_demo_dataset()
    context_string = format_dataset_as_string(dataset_list)
    question = "COUNTING: How many entries correspond to the coarse label 'NUM'?"
    
    print(f"\nDataset size: {len(dataset_list)} items")
    print(f"Context string length: {len(context_string)} characters")
    print(f"Question: {question}")
    print(f"Expected answer: 3")
    
    if args.show_data:
        print("\nContext preview (first 500 characters):")
        print(context_string[:500])
        print("...")
    
    return run_rlm(context_string, question, args)


def run_single_mode(args):
    """Run on a single example from real dataset."""
    print("="*80)
    print("MODE: SINGLE (Real TREC Data)")
    print("="*80)
    
    # Load dataset
    dataset = load_trec_dataset(args.dataset_path)
    
    # Get contexts from dataset
    contexts = dataset.get('contexts', [])
    
    if not contexts:
        print("No contexts found in dataset!")
        print("Make sure you ran the prepare_dataset.py script to generate contexts.")
        return None
    
    # Filter by context length if specified
    if args.context_length:
        filtered = [ctx for ctx in contexts if ctx.get('target_tokens') == args.context_length]
        if filtered:
            contexts = filtered
            print(f"Filtered to target token length: {args.context_length}")
        else:
            print(f"No contexts with target length {args.context_length}, using all contexts")
    
    # Select random context
    context = random.choice(contexts)
    
    # Use the pre-formatted prompt as context string
    context_string = context.get('prompt', '')
    question = context.get('final_question', 'What is the answer?')
    
    print(f"\nDataset: {len(dataset.get('augmented_examples', []))} augmented examples")
    print(f"Available contexts: {len(dataset.get('contexts', []))}")
    print(f"Selected context target tokens: {context.get('target_tokens', 'N/A')}")
    print(f"Actual example count in context: {context.get('actual_example_count', 'N/A')}")
    print(f"Context string length: {len(context_string)} characters")
    print(f"Question: {question}")
    
    if args.show_data:
        print("\nContext preview (first 1000 characters):")
        print(context_string[:1000])
        print("...")
    
    return run_rlm(context_string, question, args)


def run_batch_mode(args):
    """Run on multiple examples from real dataset."""
    print("="*80)
    print(f"MODE: BATCH ({args.num_examples} examples)")
    print("="*80)
    
    # Load dataset
    dataset = load_trec_dataset(args.dataset_path)
    
    # Get contexts from dataset
    contexts = dataset.get('contexts', [])
    
    if not contexts:
        print("No contexts found in dataset!")
        print("Make sure you ran the prepare_dataset.py script to generate contexts.")
        return None
    
    # Filter by context length if specified
    if args.context_length:
        filtered = [ctx for ctx in contexts if ctx.get('target_tokens') == args.context_length]
        if filtered:
            contexts = filtered
            print(f"Filtered to target token length: {args.context_length}")
        else:
            print(f"No contexts with target length {args.context_length}, using all contexts")
    
    # Select random contexts
    num_to_test = min(args.num_examples, len(contexts))
    selected_contexts = random.sample(contexts, num_to_test)
    
    print(f"\nTesting {num_to_test} contexts")
    
    results = []
    for i, context in enumerate(selected_contexts, 1):
        print(f"\n{'#'*80}")
        print(f"# Context {i}/{num_to_test}")
        print(f"{'#'*80}")
        
        # Use the pre-formatted prompt as context string
        context_string = context.get('prompt', '')
        question = context.get('final_question', 'What is the answer?')
        
        print(f"Question: {question}")
        print(f"Target tokens: {context.get('target_tokens', 'N/A')}")
        print(f"Example count: {context.get('actual_example_count', 'N/A')}")
        print(f"Context string length: {len(context_string)} characters")
        
        try:
            result = run_rlm(context_string, question, args)
            results.append({'success': True, 'result': result, 'question': question})
        except Exception as e:
            print(f"\nError: {str(e)}")
            results.append({'success': False, 'error': str(e), 'question': question})
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/{num_to_test}")
    print(f"Failed: {num_to_test - successful}/{num_to_test}")
    
    return results


def run_rlm(context, question, args):
    """Run RLM with given context and question."""
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
    
    print("Running RLM...\n")
    result = rlm.completion(context=context, query=question)
    
    print(f"\n{'='*80}")
    print("FINAL ANSWER:")
    print(f"{result}")
    print(f"{'='*80}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Test RLM on TREC QC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with synthetic data
  python run_rlm_on_trec.py --mode demo
  
  # Single real example with logging
  python run_rlm_on_trec.py --mode single
  
  # Test 5 examples (uses default gpt-5-mini-2025-08-07)
  python run_rlm_on_trec.py --mode batch --num-examples 5
  
  # Use Google Gemini models
  python run_rlm_on_trec.py --mode single --model gemini-1.5-pro
  python run_rlm_on_trec.py --mode single --model gemini-1.5-flash
  python run_rlm_on_trec.py --mode single --model gemini-2.0-flash-exp
  
  # Mix providers (Gemini main, GPT for sub-calls)
  python run_rlm_on_trec.py --mode single --model gemini-1.5-pro --recursive-model gpt-4o-mini
  
  # Test large context example
  python run_rlm_on_trec.py --mode single --context-length 32768 --max-iterations 20
  
  # Clean output without logging
  python run_rlm_on_trec.py --mode single --no-logging
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['demo', 'single', 'batch'],
        default='demo',
        help='demo: synthetic data (fast), single: one real example, batch: multiple examples'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset-path',
        default='/Users/parthavshergill/open_rlm/outputs/trec_qc_coarse_improved.json',
        help='Path to TREC dataset JSON file'
    )
    
    parser.add_argument(
        '--context-length',
        type=int,
        help='Filter contexts by target token length (e.g., 1024, 8192, 32768, 131072, 262144)'
    )
    
    # Batch mode options
    parser.add_argument(
        '--num-examples',
        type=int,
        default=3,
        help='Number of examples to test in batch mode (default: 3)'
    )
    
    # RLM configuration
    parser.add_argument(
        '--model',
        default='gpt-5-mini-2025-08-07',
        help='Main model to use. Supports OpenAI (gpt-5-mini-2025-08-07, gpt-4, gpt-4o, gpt-4o-mini) and Gemini (gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp) (default: gpt-5-mini-2025-08-07)'
    )
    
    parser.add_argument(
        '--recursive-model',
        default=None,
        help='Model for recursive/sub-LLM calls. Can mix providers (e.g., gemini main + gpt recursive) (default: same as --model)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=15,
        help='Maximum iterations before forcing answer (default: 15)'
    )
    
    parser.add_argument(
        '--no-logging',
        dest='logging',
        action='store_false',
        help='Disable detailed logging'
    )
    
    parser.add_argument(
        '--show-data',
        action='store_true',
        help='Show preview of context data'
    )
    
    parser.set_defaults(logging=True)
    
    args = parser.parse_args()
    
    # Set recursive model to main model if not specified
    if args.recursive_model is None:
        args.recursive_model = args.model
    
    # Check dataset exists for non-demo modes
    if args.mode != 'demo' and not Path(args.dataset_path).exists():
        print(f"Error: Dataset not found at {args.dataset_path}")
        return
    
    # Run selected mode
    if args.mode == 'demo':
        run_demo_mode(args)
    elif args.mode == 'single':
        run_single_mode(args)
    elif args.mode == 'batch':
        run_batch_mode(args)


if __name__ == "__main__":
    main()



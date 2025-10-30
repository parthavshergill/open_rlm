"""
Pipeline for preparing the TREC-QC coarse dataset following the OOLONG-synth
procedure. The script performs three high-level stages:

1. Filter out unusually hard examples using zero-shot classifications from two
   `gemini-1.5-flash` calls.
2. Augment remaining examples with synthetic user IDs and dates.
3. Construct long-context prompts (up to 256K tokens) along with question sets
   spanning counting, user, and timeline reasoning tasks.

The resulting artifacts are serialized to JSON so they can be inspected or fed
into downstream tooling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from datasets import DatasetDict, load_dataset
import google.generativeai as genai
from tqdm.auto import tqdm


DEFAULT_CONTEXT_LENGTHS = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]


def canonicalize_label(label: str) -> str:
    return label.strip().lower()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(path)


def load_existing_artifacts(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logging.debug("No existing artifact at %s", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            logging.info("Loaded existing artifact from %s", path)
            return data
    except Exception as exc:
        logging.warning("Failed to read existing artifact %s: %s", path, exc)
        return {}


def save_partial_artifacts(path: Path, artifacts: Dict[str, Any], note: Optional[str] = None) -> None:
    atomic_write_json(path, artifacts)
    if note:
        logging.info("Checkpoint saved to %s (%s)", path, note)
    else:
        logging.info("Checkpoint saved to %s", path)


def compute_dataset_digest(records: Sequence[Dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for record in records:
        serialized = json.dumps(record, sort_keys=True, ensure_ascii=False)
        hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()


def deterministic_rng(seed: int, index: int, salt: str) -> random.Random:
    base = f"{seed}:{index}:{salt}".encode("utf-8")
    digest = hashlib.sha256(base).hexdigest()
    return random.Random(int(digest, 16))


class ZeroShotClassifier:
    """Wrapper that requests coarse label predictions from Gemini models."""

    def __init__(
        self,
        model_name: str,
        label_texts: Sequence[str],
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        batch_size: int = 10,
        rate_limit_delay: float = 1.0,
    ) -> None:
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.label_lookup = {
            canonicalize_label(label_text): label_text for label_text in label_texts
        }
        self.system_prompt = (
            "You are a precise text classifier. "
            "Choose the correct coarse label for the given question."
        )
        self.user_prompt_template = (
            "You will receive a natural language question. "
            "Classify it into one of the following coarse labels: "
            f"{', '.join(sorted(self.label_lookup.values()))}. "
            "Respond ONLY with a JSON object of the form {\"label\": \"<label>\"}."
        )

    def classify(self, text: str) -> Optional[str]:
        # Combine system prompt and user prompt into a single message for Gemini
        full_prompt = f"{self.system_prompt}\n\n{self.user_prompt_template}\n\nQuestion: {text}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                    ),
                )
                logging.debug(f"Prompt: {full_prompt[:500]}...\n\nResponse: {response.text[:500]}...")
                raw = response.text.strip()
                label = self._extract_label(raw)
                if label is not None:
                    return label
            except Exception as exc:
                logging.warning(
                    "Model call failed (attempt %d/%d) for model %s: %s",
                    attempt,
                    self.max_retries,
                    self.model_name,
                    exc,
                )

            time.sleep(self.retry_delay * attempt)

        logging.error("All retries exhausted for model %s", self.model_name)
        return None

    def _extract_label(self, raw_response: str) -> Optional[str]:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return None

        label_value = parsed.get("label")
        if not isinstance(label_value, str):
            return None

        return self.label_lookup.get(canonicalize_label(label_value))

    def batch_classify(self, texts: Sequence[str]) -> List[Optional[str]]:
        """Classify multiple texts in batches with rate limiting."""
        results = [None] * len(texts)
        total_texts = len(texts)
        total_batches = (total_texts + self.batch_size - 1) // self.batch_size

        # Process in batches to avoid rate limits
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_indices = list(range(i, min(i + self.batch_size, len(texts))))
            current_batch = i // self.batch_size + 1

            logging.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_texts)} texts) for {self.model_name}")

            # Process each text in the batch individually but with rate limiting
            for j, text in enumerate(batch_texts):
                actual_index = batch_indices[j]
                logging.info(f"[{actual_index+1}/{total_texts}] Classifying with {self.model_name}: {text[:100]}...")
                result = self.classify(text)
                results[actual_index] = result
                logging.info(f"[{actual_index+1}/{total_texts}] → Predicted: {result}")

                # Rate limiting delay between individual requests
                if j < len(batch_texts) - 1:  # Don't delay after the last item in batch
                    time.sleep(self.rate_limit_delay)

            # Additional delay between batches
            if i + self.batch_size < len(texts):
                logging.debug(f"Batch {current_batch} complete. Pausing {self.retry_delay}s before next batch...")
                time.sleep(self.retry_delay)

        logging.info(f"Completed all {total_batches} batches for {self.model_name}")
        return results


def load_trec_qc_coarse(
    split: str, max_examples: Optional[int], seed: int
) -> List[Dict[str, object]]:
    logging.info("Loading dataset split '%s' (max_examples=%s)...", split, max_examples)
    dataset: DatasetDict | DatasetDict = load_dataset("SetFit/TREC-QC")
    if split not in dataset:
        raise ValueError(f"Split '{split}' not available. Options: {list(dataset.keys())}")

    hf_split = dataset[split]
    if max_examples is not None:
        hf_split = hf_split.shuffle(seed=seed).select(range(min(max_examples, len(hf_split))))

    records: List[Dict[str, object]] = []
    for row in hf_split:
        records.append(
            {
                "text": row["text"],
                "label": int(row["label"]),
                "label_text": row["label_text"],
                "label_original": row["label_original"],
                "label_coarse": int(row["label_coarse"]),
                "label_coarse_text": row["label_coarse_text"],
                "label_coarse_original": row["label_coarse_original"],
            }
        )

    logging.info("Loaded %d records from split '%s'", len(records), split)
    return records


def derive_coarse_labels(records: Sequence[Dict[str, object]]) -> List[str]:
    labels = sorted({record["label_coarse_original"] for record in records})
    return labels


def filter_examples_with_models(
    records: Sequence[Dict[str, object]],
    classifiers: Sequence[ZeroShotClassifier],
    checkpoint_callback: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    checkpoint_batch_size: int = 100,
) -> List[Dict[str, object]]:
    retained = []
    total = len(records)
    logging.info("Starting model-based filtering for %d records", total)

    # Extract all texts for batch processing
    texts = [record["text"] for record in records]

    # Process with each classifier using batch classification
    all_predictions = []
    for classifier_idx, classifier in enumerate(classifiers):
        logging.info(f"Running batch classification with classifier {classifier_idx + 1}/{len(classifiers)}")
        predictions = classifier.batch_classify(texts)
        all_predictions.append(predictions)

    progress = tqdm(records, desc="Filtering", unit="example")
    for idx, record in enumerate(progress):
        expected = record["label_coarse_original"]
        text = record["text"]
        passes_any = False

        # Collect all predictions for this example
        classifier_predictions = [predictions[idx] for predictions in all_predictions]

        # Check predictions from all classifiers for this record
        for clf_idx, prediction in enumerate(classifier_predictions):
            if prediction is not None and canonicalize_label(prediction) == canonicalize_label(expected):
                passes_any = True
                logging.debug(
                    f"Example {idx+1}/{total} MATCH - Classifier {clf_idx+1} predicted '{prediction}' == expected '{expected}' | Text: {text[:80]}..."
                )
                break
            else:
                logging.debug(
                    f"Example {idx+1}/{total} MISMATCH - Classifier {clf_idx+1} predicted '{prediction}' != expected '{expected}' | Text: {text[:80]}..."
                )

        if passes_any:
            retained.append(record)
            logging.info(f"✓ Retained example {idx+1}/{total} (label: {expected}): {text[:100]}...")
        else:
            all_preds_str = ", ".join([f"C{i+1}:{p}" for i, p in enumerate(classifier_predictions)])
            logging.info(f"✗ Filtered example {idx+1}/{total} (expected: {expected}, got: {all_preds_str}): {text[:100]}...")

        progress.set_postfix(retained=len(retained), filter_rate=f"{len(retained)/(idx+1)*100:.1f}%")

        # Save checkpoint after every checkpoint_batch_size examples
        if checkpoint_callback and (idx + 1) % checkpoint_batch_size == 0:
            checkpoint_callback(retained)
            logging.info(f"Checkpoint saved after processing {idx+1}/{total} examples")

    # Final checkpoint save
    if checkpoint_callback:
        checkpoint_callback(retained)
        logging.info("Final filtering checkpoint saved")

    logging.info("Filtering complete. Retained %d/%d records", len(retained), total)
    return retained


def sample_user_id(rng, pool_size: int = 1000) -> str:
    common_bucket = max(1, int(pool_size * 0.2))
    if rng.random() < 0.8:
        return f"U{rng.randint(0, common_bucket - 1)}"
    return f"U{rng.randint(0, pool_size - 1)}"


def sample_date(rng, start_date: datetime, months: int = 40) -> str:
    approximate_days = months * 30
    delta_days = rng.randint(0, approximate_days)
    sampled = start_date + timedelta(days=delta_days)
    return sampled.strftime("%m/%d/%Y")


def augment_examples(
    records: Sequence[Dict[str, object]],
    seed: int,
    checkpoint_callback: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    checkpoint_batch_size: int = 100,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    start_date = datetime(2022, 1, 1)
    augmented = []
    logging.info("Augmenting %d records with synthetic metadata", len(records))
    for idx, record in enumerate(tqdm(records, desc="Augmenting", unit="example")):
        enriched = dict(record)
        enriched["user_id"] = sample_user_id(rng)
        enriched["date"] = sample_date(rng, start_date=start_date)
        augmented.append(enriched)

        # Save checkpoint after every checkpoint_batch_size examples
        if checkpoint_callback and (idx + 1) % checkpoint_batch_size == 0:
            checkpoint_callback(augmented)
            logging.info(f"Augmentation checkpoint saved after processing {idx+1}/{len(records)} examples")

    # Final checkpoint save
    if checkpoint_callback:
        checkpoint_callback(augmented)
        logging.info("Final augmentation checkpoint saved")

    logging.info("Augmentation complete")
    return augmented


def approximate_token_count(text: str) -> int:
    # Rough heuristic: word count plus buffer for metadata.
    words = text.split()
    return max(1, int(len(words) * 1.3) + 20)


def estimate_examples_needed(instances: Sequence[Dict[str, object]], target_tokens: int) -> int:
    if not instances:
        return 0
    avg_tokens = sum(approximate_token_count(item["text"]) for item in instances) / len(instances)
    return max(1, int((0.95 * target_tokens) / max(avg_tokens, 1)))


def sample_to_match_label_distribution(
    instances: Sequence[Dict[str, object]],
    desired_count: int,
    seed: int,
) -> List[Dict[str, object]]:
    if desired_count <= 0:
        return []

    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for instance in instances:
        buckets[instance["label_coarse_original"]].append(instance)

    labels = sorted(buckets.keys())
    gamma_samples = [rng.gammavariate(1.0, 1.0) for _ in labels]
    total_gamma = sum(gamma_samples)
    proportions = [sample / total_gamma for sample in gamma_samples]

    scaled = [desired_count * proportion for proportion in proportions]
    counts = [int(math.floor(val)) for val in scaled]
    remainder = desired_count - sum(counts)

    if remainder > 0:
        fractional_pairs = sorted(
            enumerate([val - math.floor(val) for val in scaled]), key=lambda item: item[1], reverse=True
        )
        for idx, _ in fractional_pairs[:remainder]:
            counts[idx] += 1
    elif remainder < 0:
        for _ in range(-remainder):
            target_index = rng.randrange(len(counts))
            if counts[target_index] > 0:
                counts[target_index] -= 1

    sampled: List[Dict[str, object]] = []
    for label, count in zip(labels, counts):
        if count <= 0:
            continue

        population = buckets[label]
        for _ in range(count):
            sampled.append(rng.choice(population))

    rng.shuffle(sampled)
    return sampled


def generate_questions(
    instances: Sequence[Dict[str, object]],
    total_questions: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    labels = sorted({item["label_coarse_original"] for item in instances})
    user_counts = Counter(item["user_id"] for item in instances)
    if len(labels) == 0:
        return []

    def random_date_threshold() -> str:
        if not instances:
            return "01/01/2023"
        parsed_dates = [datetime.strptime(item["date"], "%m/%d/%Y") for item in instances]
        earliest = min(parsed_dates)
        latest = max(parsed_dates)
        span_days = (latest - earliest).days or 1
        offset = rng.randint(0, span_days)
        threshold = earliest + timedelta(days=offset)
        return threshold.strftime("%m/%d/%Y")

    questions: List[str] = []
    generators = ["counting", "user", "timeline"]

    while len(questions) < total_questions:
        choice = rng.choice(generators)
        if choice == "counting":
            target_label = rng.choice(labels)
            variant = rng.random()
            if variant < 0.5:
                questions.append(
                    f"COUNTING: Which coarse label appears most often in the context?"
                )
            else:
                questions.append(
                    f"COUNTING: How many entries correspond to the coarse label '{target_label}'?"
                )
        elif choice == "user":
            if user_counts:
                questions.append("USER: Which user ID occurs most frequently in the context?")
            else:
                questions.append("USER: Identify the user ID that appears the most.")
        else:
            threshold = random_date_threshold()
            focus_label = rng.choice(labels)
            questions.append(
                "TIMELINE: Compare how often the coarse label "
                f"'{focus_label}' appears before {threshold} versus on/after that date."
            )

    return questions


def build_context_prompt(
    instances: Sequence[Dict[str, object]],
    label_catalog: Sequence[str],
    final_question: str,
) -> str:
    header_lines = [
        "SYSTEM INSTRUCTIONS:",
        (
            "You are reviewing user questions over a set of examples for coarse-grained topic classification. "
            "The set of possible coarse labels are: " + ", ".join(label_catalog) + ". "
            "IMPORTANT: The instances below are NOT pre-labeled. You must semantically understand each instance "
            "to determine which coarse label it belongs to. For example, 'What ocean borders France?' should be "
            "classified as LOC (location), 'Who invented the telephone?' as HUM (human), etc. "
            "Use the llm_query tool to batch-classify instances semantically before answering counting or filtering questions."
        ),
        f"There are {len(instances)} examples in the context.",
        "Answer the question that follows.",
        "",
        "=== CONTEXT ENTRIES ===",
    ]

    entry_lines = []
    for idx, instance in enumerate(instances, start=1):
        entry_lines.append(
            f"Date: {instance['date']} || User: {instance['user_id']} || Instance: {instance['text']}"
        )

    footer_lines = [
        "",
        "END OF CONTEXT. Answer the following question without restating the context:",
        final_question,
    ]

    return "\n".join(header_lines + entry_lines + footer_lines)


def construct_context_windows(
    augmented_instances: Sequence[Dict[str, object]],
    label_catalog: Sequence[str],
    context_lengths: Sequence[int],
    seed: int,
    checkpoint_callback: Optional[Callable[[List[Dict[str, object]]], None]] = None,
) -> List[Dict[str, object]]:
    contexts: List[Dict[str, object]] = []
    logging.info("Constructing %d context windows", len(context_lengths))

    for idx, target_tokens in enumerate(tqdm(context_lengths, desc="Contexts", unit="target"), start=1):
        base_rng = deterministic_rng(seed, idx, "context")
        desired_examples = estimate_examples_needed(augmented_instances, target_tokens)
        final_instances = sample_to_match_label_distribution(
            augmented_instances, desired_examples, seed=base_rng.randint(0, 10**9)
        )
        question_seed = base_rng.randint(0, 10**9)
        questions = generate_questions(final_instances, total_questions=25, seed=question_seed)
        final_question = base_rng.choice(questions) if questions else "COUNTING: No question generated."
        prompt = build_context_prompt(final_instances, label_catalog, final_question)

        context_window = {
            "target_tokens": target_tokens,
            "requested_example_count": desired_examples,
            "actual_example_count": len(final_instances),
            "prompt": prompt,
            "questions": questions,
            "final_question": final_question,
        }
        contexts.append(context_window)

        # Save checkpoint after each context window
        if checkpoint_callback:
            checkpoint_callback(contexts)
            logging.info(f"Context construction checkpoint saved after building context window {idx}/{len(context_lengths)}")

    return contexts


def save_artifacts(
    output_path: Path,
    metadata: Dict[str, object],
    augmented_examples: Sequence[Dict[str, object]],
    contexts: Sequence[Dict[str, object]],
) -> None:
    output = {
        "metadata": metadata,
        "augmented_examples": augmented_examples,
        "contexts": contexts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, output)
    logging.info("Saved prepared dataset to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TREC-QC coarse dataset")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument(
        "--output",
        default="./outputs/trec_qc_coarse_prepared.json",
        help="Path to save the resulting JSON artifacts",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="*",
        default=DEFAULT_CONTEXT_LENGTHS,
        help="Target context lengths (tokens) to construct",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional maximum number of raw examples to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for augmentation and sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for zero-shot classifier requests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt to resume from the output file if it already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(verbose=args.verbose)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY must be set before running this script.")

    # Configure Gemini API
    genai.configure(api_key=api_key)

    output_path = Path(args.output)
    existing = load_existing_artifacts(output_path) if args.resume else {}

    logging.info("Loading TREC-QC coarse dataset split '%s'...", args.split)
    raw_records = load_trec_qc_coarse(args.split, args.max_examples, args.seed)
    label_catalog = derive_coarse_labels(raw_records)
    logging.info("Dataset digest: %s", compute_dataset_digest(raw_records))
    logging.info("Loaded %d examples with %d coarse labels.", len(raw_records), len(label_catalog))

    classifiers = [
        ZeroShotClassifier(
            model_name="gemini-flash-lite-latest",
            label_texts=label_catalog,
            temperature=args.temperature,
            batch_size=100,  # Process 100 requests at a time
            rate_limit_delay=1.0,  # 1 second delay between requests
        ),
        ZeroShotClassifier(
            model_name="gemini-flash-lite-latest",
            label_texts=label_catalog,
            temperature=args.temperature + 0.5,
            batch_size=100,  # Process 100 requests at a time
            rate_limit_delay=1.0,  # 1 second delay between requests
        ),
    ]

    # Define checkpoint callback functions
    def create_filtering_checkpoint_callback():
        def checkpoint_callback(validated_records):
            partial_metadata = existing.get("metadata", {})
            partial_metadata.update(
                {
                    "split": args.split,
                    "context_lengths": args.context_lengths,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "model_names": [classifier.model_name for classifier in classifiers],
                    "total_raw_examples": len(raw_records),
                    "validated_examples": len(validated_records),
                }
            )
            save_partial_artifacts(
                output_path,
                {
                    "metadata": partial_metadata,
                    "validated_examples": validated_records,
                },
                note="filtering-checkpoint",
            )
        return checkpoint_callback

    def create_augmentation_checkpoint_callback(validated_records):
        def checkpoint_callback(augmented_records):
            partial_metadata = existing.get("metadata", {})
            partial_metadata.update(
                {
                    "split": args.split,
                    "context_lengths": args.context_lengths,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "model_names": [classifier.model_name for classifier in classifiers],
                    "total_raw_examples": len(raw_records),
                    "validated_examples": len(validated_records),
                }
            )
            save_partial_artifacts(
                output_path,
                {
                    "metadata": partial_metadata,
                    "validated_examples": validated_records,
                    "augmented_examples": augmented_records,
                },
                note="augmentation-checkpoint",
            )
        return checkpoint_callback

    def create_context_checkpoint_callback(validated_records, augmented_records):
        def checkpoint_callback(contexts):
            partial_metadata = existing.get("metadata", {})
            partial_metadata.update(
                {
                    "split": args.split,
                    "context_lengths": args.context_lengths,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "model_names": [classifier.model_name for classifier in classifiers],
                    "total_raw_examples": len(raw_records),
                    "validated_examples": len(validated_records),
                }
            )
            save_partial_artifacts(
                output_path,
                {
                    "metadata": partial_metadata,
                    "validated_examples": validated_records,
                    "augmented_examples": augmented_records,
                    "contexts": contexts,
                },
                note="contexts-checkpoint",
            )
        return checkpoint_callback

    validated_records: List[Dict[str, Any]]
    if existing.get("validated_examples"):
        logging.info(
            "Resuming with %d validated records from artifact",
            len(existing["validated_examples"]),
        )
        validated_records = existing["validated_examples"]
    else:
        logging.info("Starting model-based filtering with incremental checkpoints")
        validated_records = filter_examples_with_models(
            raw_records,
            classifiers,
            checkpoint_callback=create_filtering_checkpoint_callback(),
            checkpoint_batch_size=50,  # Save every 50 examples
        )

    augmented_records: List[Dict[str, Any]]
    if existing.get("augmented_examples"):
        logging.info(
            "Resuming with %d augmented records from artifact",
            len(existing["augmented_examples"]),
        )
        augmented_records = existing["augmented_examples"]
    else:
        logging.info("Starting data augmentation with incremental checkpoints")
        augmented_records = augment_examples(
            validated_records,
            seed=args.seed,
            checkpoint_callback=create_augmentation_checkpoint_callback(validated_records),
            checkpoint_batch_size=100,  # Save every 100 examples
        )

    constructed_contexts: List[Dict[str, Any]]
    if existing.get("contexts"):
        logging.info("Resuming with %d contexts from artifact", len(existing["contexts"]))
        constructed_contexts = existing["contexts"]
    else:
        logging.info("Starting context construction with incremental checkpoints")
        constructed_contexts = construct_context_windows(
            augmented_records,
            label_catalog,
            args.context_lengths,
            seed=args.seed,
            checkpoint_callback=create_context_checkpoint_callback(validated_records, augmented_records),
        )

    metadata = {
        "split": args.split,
        "total_raw_examples": len(raw_records),
        "validated_examples": len(validated_records),
        "context_lengths": args.context_lengths,
        "temperature": args.temperature,
        "seed": args.seed,
        "model_names": [classifier.model_name for classifier in classifiers],
        "dataset_digest": compute_dataset_digest(raw_records),
    }

    save_artifacts(output_path, metadata, augmented_records, constructed_contexts)


if __name__ == "__main__":
    main()
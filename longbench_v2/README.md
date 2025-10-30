# LongBench-v2 Evaluation for RLM

This folder contains scripts for evaluating the Recursive Language Model (RLM) on the [LongBench-v2 benchmark](https://github.com/THUDM/LongBench).

## Overview

LongBench-v2 is a benchmark designed to assess LLMs' ability to handle long-context problems requiring **deep understanding and reasoning** across real-world multitasks. It consists of 503 challenging multiple-choice questions with contexts ranging from 8k to 2M words across six major task categories:

1. **Single-document QA**: Question answering on individual long documents
2. **Multi-document QA**: Question answering across multiple documents
3. **Long in-context learning**: Learning from many examples in context
4. **Long-dialogue history understanding**: Understanding long conversation histories
5. **Code repository understanding**: Understanding large codebases
6. **Long structured data understanding**: Processing large structured datasets

## Setup

### Install Dependencies

First, make sure you have the required dependencies:

```bash
cd ~/open_rlm/
source .venv/bin/activate
pip install datasets
```

The `datasets` library from Hugging Face is required to download LongBench-v2.

## Usage

### Step 1: Download the Dataset

First, download the LongBench-v2 dataset:

```bash
cd ~/open_rlm/longbench-eval
source ../.venv/bin/activate

# Download full dataset
python load_dataset.py --output datasets/longbench_v2.json

# Or download filtered subsets
python load_dataset.py --output datasets/easy.json --filter-difficulty easy
python load_dataset.py --output datasets/short.json --filter-length short
python load_dataset.py --output datasets/single_doc.json --filter-domain single_document_qa

# Download a small sample for testing
python load_dataset.py --output datasets/test_sample.json --max-samples 50
```

### Step 2: Run RLM Evaluation

Run RLM on the dataset:

```bash
# Evaluate on a single example
python run_rlm_on_longbench.py --mode single --dataset datasets/longbench_v2.json

# Evaluate on multiple examples
python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --num-examples 10

# Use specific model
python run_rlm_on_longbench.py --mode single --dataset datasets/longbench_v2.json --model gpt-4o

# Use Gemini models
python run_rlm_on_longbench.py --mode single --dataset datasets/longbench_v2.json --model gemini-1.5-pro

# Filter by domain or difficulty during evaluation
python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --filter-domain single_document_qa --num-examples 5

# Save results to file
python run_rlm_on_longbench.py --mode batch --dataset datasets/longbench_v2.json --num-examples 20 --output-results results/eval_1.json
```

## Dataset Format

Each LongBench-v2 example has the following structure:

```json
{
    "_id": "unique_identifier",
    "domain": "primary_domain_category",
    "sub_domain": "specific_sub_domain",
    "difficulty": "easy or hard",
    "length": "short, medium, or long",
    "question": "The question text",
    "choice_A": "Option A",
    "choice_B": "Option B",
    "choice_C": "Option C",
    "choice_D": "Option D",
    "answer": "A, B, C, or D",
    "context": "The long context (8k-2M words)"
}
```

## Results

Results are saved in JSON format with the following structure:

```json
{
    "metadata": {
        "timestamp": "2025-10-30T...",
        "dataset": "path/to/dataset.json",
        "model": "gpt-5-mini-2025-08-07",
        "num_examples": 10,
        "accuracy": 75.5,
        "filters": {...}
    },
    "results": [
        {
            "item_id": "...",
            "domain": "...",
            "correct_answer": "A",
            "extracted_answer": "A",
            "is_correct": true,
            "model_response": "..."
        },
        ...
    ]
}
```

## Command Reference

### Load Dataset

```bash
# Full dataset
python load_dataset.py --output datasets/longbench_v2.json

# Filter options
python load_dataset.py --output <path> --filter-domain <domain>
python load_dataset.py --output <path> --filter-difficulty <easy|hard>
python load_dataset.py --output <path> --filter-length <short|medium|long>
python load_dataset.py --output <path> --max-samples <number>
```

### Run Evaluation

```bash
# Single example
python run_rlm_on_longbench.py --mode single --dataset <path>

# Batch evaluation
python run_rlm_on_longbench.py --mode batch --dataset <path> --num-examples <n>

# Model selection
python run_rlm_on_longbench.py --model <model_name> --dataset <path>

# Filter during evaluation
python run_rlm_on_longbench.py --filter-domain <domain> --dataset <path>
python run_rlm_on_longbench.py --filter-difficulty <easy|hard> --dataset <path>

# Save results
python run_rlm_on_longbench.py --output-results <path> --dataset <path>
```

## Notes

- The RLM uses the same architecture as in the TREC evaluation, with recursive sub-calls at depth=1
- Questions are formatted as multiple-choice with options A, B, C, D
- The RLM extracts the answer choice from the model's response
- Accuracy is calculated across all successful evaluations
- Results include breakdowns by domain, difficulty, and length when available

## Citation

If you use LongBench-v2 in your research, please cite:

```bibtex
@article{bai2024longbench2,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks}, 
  author={Yushi Bai and Shangqing Tu and Jiajie Zhang and Hao Peng and Xiaozhi Wang and Xin Lv and Shulin Cao and Jiazheng Xu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2412.15204},
  year={2024}
}
```


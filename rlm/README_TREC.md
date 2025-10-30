# Testing RLM on TREC Dataset

One configurable script to test RLM on the TREC Question Classification dataset.

## Quick Start

```bash
# Demo with synthetic data (fastest way to understand)
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode demo

# Single real example
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode single

# Multiple examples
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode batch --num-examples 5
```

## All Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--mode` | demo, single, batch | demo | Test mode to run |
| `--num-examples` | integer | 3 | Number of examples (batch mode) |
| `--context-length` | 1024, 8192, 32768, etc. | - | Filter by context size |
| `--model` | gpt-4, gpt-4o-mini, etc. | gpt-4 | Main reasoning model |
| `--recursive-model` | gpt-4, gpt-4o-mini, etc. | same as model | Model for sub-calls |
| `--max-iterations` | integer | 15 | Max iterations before forcing answer |
| `--no-logging` | flag | - | Disable detailed logging |
| `--show-data` | flag | - | Show context preview |
| `--dataset-path` | path | (has default) | Path to TREC JSON file |

## Examples

### Use cheaper model
```bash
python run_rlm_on_trec.py --mode single --model gpt-4o-mini
```

### Test large context (262K tokens)
```bash
python run_rlm_on_trec.py --mode single --context-length 262144 --max-iterations 20
```

### Clean output
```bash
python run_rlm_on_trec.py --mode single --no-logging
```

### Batch test with specific context length
```bash
python run_rlm_on_trec.py --mode batch --num-examples 10 --context-length 8192 --model gpt-4o-mini
```

### See help
```bash
python run_rlm_on_trec.py --help
```

## What the Modes Do

**demo**: Uses 8 synthetic examples, fast, great for understanding RLM mechanics

**single**: Picks one random example from real TREC dataset, shows full reasoning

**batch**: Tests multiple examples, provides success statistics

## Dataset Info

The TREC dataset contains:
- 3,630 validated question-answer pairs
- Context lengths: 1K to 262K tokens
- Query types: COUNTING, USER, TIMELINE, etc.

RLM handles these well because it can write Python code for exact counting/filtering and use recursive calls for analysis.



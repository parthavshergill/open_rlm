# 🚀 Quick Commands to Test RLM on TREC Dataset

One script, multiple modes - just change the parameters!

## Quick Start Commands

### 1. Demo Mode (Synthetic Data - FASTEST!)
```bash
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode demo
```
⏱️ **Time**: ~30 seconds  
📝 Uses 8 synthetic examples to show how RLM works

---

### 2. Single Example (Real TREC Data)
```bash
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode single
```
⏱️ **Time**: 1-3 minutes  
📝 Tests one random example from real dataset

---

### 3. Batch Testing (Multiple Examples)
```bash
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode batch --num-examples 5
```
⏱️ **Time**: 5-15 minutes  
📝 Tests multiple examples with statistics

---

## Common Parameter Combinations

### Use cheaper/faster model
```bash
python run_rlm_on_trec.py --mode single --model gpt-4o-mini --recursive-model gpt-4o-mini
```

### Test large context (262K tokens)
```bash
python run_rlm_on_trec.py --mode single --context-length 262144 --max-iterations 20
```

### Clean output (no logging)
```bash
python run_rlm_on_trec.py --mode single --no-logging
```

### See the data structure
```bash
python run_rlm_on_trec.py --mode demo --show-data
```

---

## All Available Parameters

```
--mode              demo | single | batch (default: demo)
--num-examples      Number of examples in batch mode (default: 3)
--context-length    Filter by context size: 1024, 8192, 32768, etc.
--model             Main model (default: gpt-4)
--recursive-model   Model for sub-calls (default: same as --model)
--max-iterations    Max iterations (default: 15)
--no-logging        Disable detailed logging
--show-data         Show preview of context
--dataset-path      Path to dataset (has default)
```

---

## ⚙️ Before Running: Check Your Setup

```bash
# 1. Verify API key is set
cat ~/open_rlm/.env.local | grep OPENAI_API_KEY

# 2. Verify dataset exists
ls -lh ~/open_rlm/outputs/trec_qc_coarse_prepared.json

# 3. Verify dependencies
cd ~/open_rlm/rlm && pip list | grep -E "openai|rich|dotenv"
```

---

## 🎯 Recommended Path

**Step 1**: Start with demo mode
```bash
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode demo
```

**Step 2**: Try a real example
```bash
cd ~/open_rlm && dotenvx run -f .env.local python ~/open_rlm/rlm/run_rlm_on_trec.py --mode single
```

**Step 3**: Experiment with parameters
```bash
# Try different models, context sizes, etc.
python run_rlm_on_trec.py --mode single --model gpt-4o-mini --context-length 8192
```

**Step 4**: See help for all options
```bash
python run_rlm_on_trec.py --help
```

---

## 📖 More Info

See `rlm/README_TREC.md` for detailed parameter documentation and examples.


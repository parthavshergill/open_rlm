#!/bin/bash

# Quick start script for LongBench-v2 evaluation
# This script helps you get started with RLM evaluation on LongBench-v2

set -e

echo "=========================================="
echo "LongBench-v2 RLM Evaluation - Quick Start"
echo "=========================================="
echo ""

# Navigate to project root
cd ~/open_rlm/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Navigate to longbench-eval folder
cd longbench-eval

# Check if datasets library is installed
if ! python -c "import datasets" 2>/dev/null; then
    echo "Installing datasets library..."
    pip install datasets
fi

# Create directories
mkdir -p datasets
mkdir -p results

echo ""
echo "Setup complete!"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Download a sample dataset (recommended for first time):"
echo "   python load_dataset.py --output datasets/test_sample.json --max-samples 10"
echo ""
echo "2. Run a single evaluation:"
echo "   python run_rlm_on_longbench.py --mode single --dataset datasets/test_sample.json --show-question"
echo ""
echo "3. Run batch evaluation:"
echo "   python run_rlm_on_longbench.py --mode batch --dataset datasets/test_sample.json --num-examples 3 --output-results results/test_run.json"
echo ""
echo "4. Download full dataset:"
echo "   python load_dataset.py --output datasets/longbench_v2.json"
echo ""
echo "5. See README.md for more options and examples"
echo ""


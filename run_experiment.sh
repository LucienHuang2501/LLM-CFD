#!/bin/bash
# LLM-CFD Experiment Runner
# 
# Usage:
#   1. Set your DeepSeek API key:
#      export DEEPSEEK_API_KEY="sk-your-key-here"
#      
#   2. Run:
#      bash run_experiment.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "LLM-CFD Experiment Suite"
echo "============================================"
echo "Working directory: $(pwd)"
echo ""

# Check API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set."
    echo ""
    echo "Please set it first:"
    echo "  export DEEPSEEK_API_KEY='sk-your-key-here'"
    echo ""
    exit 1
fi

echo "DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:0:10}..."
echo ""

# Find working Python (with required packages)
PYTHON=""
for py in /usr/local/bin/python3 /usr/bin/python3 python3; do
    if $py -c "from openai import OpenAI" 2>/dev/null; then
        PYTHON="$py"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Installing openai..."
    python3 -m pip install openai --quiet
    # Re-check after install
    for py in /usr/local/bin/python3 /usr/bin/python3 python3; do
        if $py -c "from openai import OpenAI" 2>/dev/null; then
            PYTHON="$py"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "ERROR: Cannot find Python with openai installed."
    echo "Please run: python3 -m pip install openai pandas scikit-learn scipy"
    exit 1
fi

echo "Using Python: $PYTHON"

# Create directories
mkdir -p data results results/cache

# Run experiment
$PYTHON src/experiment.py

echo ""
echo "Experiment complete! Results in: results/"

#!/bin/bash

# COMPLETE BIAS EXPERIMENT SUITE
# Generated on 2025-06-29 20:03:59.462558
# Total conditions: 137

echo "Starting complete bias experiment suite (154 conditions)"

echo "=== BASELINE EXPERIMENTS ==="
./run_baseline_20250629_200359.sh

echo "=== PROMPT_ONLY EXPERIMENTS ==="
./run_prompt_only_20250629_200359.sh

echo "=== CLAMPING_ONLY EXPERIMENTS ==="
./run_clamping_only_20250629_200359.sh

echo "=== BOTH EXPERIMENTS ==="
./run_both_20250629_200359.sh

echo "=== EQUIVALENCE EXPERIMENTS ==="
./run_equivalence_20250629_200359.sh


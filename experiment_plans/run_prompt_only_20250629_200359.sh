#!/bin/bash

# PROMPT_ONLY EXPERIMENTS
# Generated on 2025-06-29 20:03:59.462112

echo "Running prompt_only experiment 1/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric --output-suffix prompt_only_pediatric

echo "Running prompt_only experiment 2/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent --output-suffix prompt_only_adolescent

echo "Running prompt_only experiment 3/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult --output-suffix prompt_only_young_adult

echo "Running prompt_only experiment 4/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age --output-suffix prompt_only_middle_age

echo "Running prompt_only experiment 5/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior --output-suffix prompt_only_senior

echo "Running prompt_only experiment 6/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt male --output-suffix prompt_only_male

echo "Running prompt_only experiment 7/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt female --output-suffix prompt_only_female

echo "Running prompt_only experiment 8/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric male --output-suffix prompt_only_pediatric_male

echo "Running prompt_only experiment 9/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric female --output-suffix prompt_only_pediatric_female

echo "Running prompt_only experiment 10/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent male --output-suffix prompt_only_adolescent_male

echo "Running prompt_only experiment 11/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent female --output-suffix prompt_only_adolescent_female

echo "Running prompt_only experiment 12/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult male --output-suffix prompt_only_young_adult_male

echo "Running prompt_only experiment 13/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult female --output-suffix prompt_only_young_adult_female

echo "Running prompt_only experiment 14/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age male --output-suffix prompt_only_middle_age_male

echo "Running prompt_only experiment 15/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age female --output-suffix prompt_only_middle_age_female

echo "Running prompt_only experiment 16/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior male --output-suffix prompt_only_senior_male

echo "Running prompt_only experiment 17/17"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior female --output-suffix prompt_only_senior_female


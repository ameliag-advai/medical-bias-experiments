#!/bin/bash

# EQUIVALENCE EXPERIMENTS
# Generated on 2025-06-29 20:03:59.462439

echo "Running equivalence experiment 1/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric --output-suffix equiv_prompt_pediatric

echo "Running equivalence experiment 2/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features pediatric --clamp-intensity 1.0 --output-suffix equiv_clamp_pediatric

echo "Running equivalence experiment 3/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent --output-suffix equiv_prompt_adolescent

echo "Running equivalence experiment 4/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features adolescent --clamp-intensity 1.0 --output-suffix equiv_clamp_adolescent

echo "Running equivalence experiment 5/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult --output-suffix equiv_prompt_young_adult

echo "Running equivalence experiment 6/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features young_adult --clamp-intensity 1.0 --output-suffix equiv_clamp_young_adult

echo "Running equivalence experiment 7/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age --output-suffix equiv_prompt_middle_age

echo "Running equivalence experiment 8/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features middle_age --clamp-intensity 1.0 --output-suffix equiv_clamp_middle_age

echo "Running equivalence experiment 9/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior --output-suffix equiv_prompt_senior

echo "Running equivalence experiment 10/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features senior --clamp-intensity 1.0 --output-suffix equiv_clamp_senior

echo "Running equivalence experiment 11/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt male --output-suffix equiv_prompt_male

echo "Running equivalence experiment 12/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features male --clamp-intensity 1.0 --output-suffix equiv_clamp_male

echo "Running equivalence experiment 13/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt female --output-suffix equiv_prompt_female

echo "Running equivalence experiment 14/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features female --clamp-intensity 1.0 --output-suffix equiv_clamp_female

echo "Running equivalence experiment 15/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric male --output-suffix equiv_prompt_pediatric_male

echo "Running equivalence experiment 16/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features pediatric male --clamp-intensity 1.0 --output-suffix equiv_clamp_pediatric_male

echo "Running equivalence experiment 17/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric female --output-suffix equiv_prompt_pediatric_female

echo "Running equivalence experiment 18/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features pediatric female --clamp-intensity 1.0 --output-suffix equiv_clamp_pediatric_female

echo "Running equivalence experiment 19/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent male --output-suffix equiv_prompt_adolescent_male

echo "Running equivalence experiment 20/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features adolescent male --clamp-intensity 1.0 --output-suffix equiv_clamp_adolescent_male

echo "Running equivalence experiment 21/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent female --output-suffix equiv_prompt_adolescent_female

echo "Running equivalence experiment 22/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features adolescent female --clamp-intensity 1.0 --output-suffix equiv_clamp_adolescent_female

echo "Running equivalence experiment 23/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult male --output-suffix equiv_prompt_young_adult_male

echo "Running equivalence experiment 24/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features young_adult male --clamp-intensity 1.0 --output-suffix equiv_clamp_young_adult_male

echo "Running equivalence experiment 25/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult female --output-suffix equiv_prompt_young_adult_female

echo "Running equivalence experiment 26/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features young_adult female --clamp-intensity 1.0 --output-suffix equiv_clamp_young_adult_female

echo "Running equivalence experiment 27/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age male --output-suffix equiv_prompt_middle_age_male

echo "Running equivalence experiment 28/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features middle_age male --clamp-intensity 1.0 --output-suffix equiv_clamp_middle_age_male

echo "Running equivalence experiment 29/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age female --output-suffix equiv_prompt_middle_age_female

echo "Running equivalence experiment 30/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features middle_age female --clamp-intensity 1.0 --output-suffix equiv_clamp_middle_age_female

echo "Running equivalence experiment 31/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior male --output-suffix equiv_prompt_senior_male

echo "Running equivalence experiment 32/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features senior male --clamp-intensity 1.0 --output-suffix equiv_clamp_senior_male

echo "Running equivalence experiment 33/34"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior female --output-suffix equiv_prompt_senior_female

echo "Running equivalence experiment 34/34"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features senior female --clamp-intensity 1.0 --output-suffix equiv_clamp_senior_female


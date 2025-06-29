#!/bin/bash

# EQUIVALENCE VALIDATION TESTS
# Generated: 2025-06-29 20:12:42.820195
# Total commands: 40

echo "🧪 Starting Equivalence Validation Tests"
echo "📊 Total tests: 40"

echo "[1/40] Full prompt: middle_age female"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age female --output-suffix equiv_full_prompt_middle_age_female_20250629_201242
echo "✅ Test completed"

echo "[2/40] Age prompt (middle_age) + sex clamping (female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age --clamp-features female --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_middle_age_female_20250629_201242
echo "✅ Test completed"

echo "[3/40] Sex prompt (female) + age clamping (middle_age 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt female --clamp-features middle_age --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_middle_age_female_20250629_201242
echo "✅ Test completed"

echo "[4/40] Neutral prompt + both clamping (middle_age female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features middle_age female --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_middle_age_female_20250629_201242
echo "✅ Test completed"

echo "[5/40] Full prompt: pediatric male"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric male --output-suffix equiv_full_prompt_pediatric_male_20250629_201242
echo "✅ Test completed"

echo "[6/40] Age prompt (pediatric) + sex clamping (male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric --clamp-features male --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_pediatric_male_20250629_201242
echo "✅ Test completed"

echo "[7/40] Sex prompt (male) + age clamping (pediatric 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt male --clamp-features pediatric --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_pediatric_male_20250629_201242
echo "✅ Test completed"

echo "[8/40] Neutral prompt + both clamping (pediatric male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features pediatric male --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_pediatric_male_20250629_201242
echo "✅ Test completed"

echo "[9/40] Full prompt: middle_age male"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age male --output-suffix equiv_full_prompt_middle_age_male_20250629_201242
echo "✅ Test completed"

echo "[10/40] Age prompt (middle_age) + sex clamping (male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt middle_age --clamp-features male --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_middle_age_male_20250629_201242
echo "✅ Test completed"

echo "[11/40] Sex prompt (male) + age clamping (middle_age 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt male --clamp-features middle_age --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_middle_age_male_20250629_201242
echo "✅ Test completed"

echo "[12/40] Neutral prompt + both clamping (middle_age male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features middle_age male --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_middle_age_male_20250629_201242
echo "✅ Test completed"

echo "[13/40] Full prompt: senior female"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior female --output-suffix equiv_full_prompt_senior_female_20250629_201242
echo "✅ Test completed"

echo "[14/40] Age prompt (senior) + sex clamping (female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior --clamp-features female --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_senior_female_20250629_201242
echo "✅ Test completed"

echo "[15/40] Sex prompt (female) + age clamping (senior 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt female --clamp-features senior --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_senior_female_20250629_201242
echo "✅ Test completed"

echo "[16/40] Neutral prompt + both clamping (senior female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features senior female --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_senior_female_20250629_201242
echo "✅ Test completed"

echo "[17/40] Full prompt: senior male"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior male --output-suffix equiv_full_prompt_senior_male_20250629_201242
echo "✅ Test completed"

echo "[18/40] Age prompt (senior) + sex clamping (male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt senior --clamp-features male --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_senior_male_20250629_201242
echo "✅ Test completed"

echo "[19/40] Sex prompt (male) + age clamping (senior 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt male --clamp-features senior --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_senior_male_20250629_201242
echo "✅ Test completed"

echo "[20/40] Neutral prompt + both clamping (senior male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features senior male --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_senior_male_20250629_201242
echo "✅ Test completed"

echo "[21/40] Full prompt: young_adult male"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult male --output-suffix equiv_full_prompt_young_adult_male_20250629_201242
echo "✅ Test completed"

echo "[22/40] Age prompt (young_adult) + sex clamping (male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult --clamp-features male --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_young_adult_male_20250629_201242
echo "✅ Test completed"

echo "[23/40] Sex prompt (male) + age clamping (young_adult 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt male --clamp-features young_adult --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_young_adult_male_20250629_201242
echo "✅ Test completed"

echo "[24/40] Neutral prompt + both clamping (young_adult male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features young_adult male --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_young_adult_male_20250629_201242
echo "✅ Test completed"

echo "[25/40] Full prompt: pediatric female"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric female --output-suffix equiv_full_prompt_pediatric_female_20250629_201242
echo "✅ Test completed"

echo "[26/40] Age prompt (pediatric) + sex clamping (female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt pediatric --clamp-features female --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_pediatric_female_20250629_201242
echo "✅ Test completed"

echo "[27/40] Sex prompt (female) + age clamping (pediatric 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt female --clamp-features pediatric --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_pediatric_female_20250629_201242
echo "✅ Test completed"

echo "[28/40] Neutral prompt + both clamping (pediatric female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features pediatric female --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_pediatric_female_20250629_201242
echo "✅ Test completed"

echo "[29/40] Full prompt: young_adult female"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult female --output-suffix equiv_full_prompt_young_adult_female_20250629_201242
echo "✅ Test completed"

echo "[30/40] Age prompt (young_adult) + sex clamping (female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt young_adult --clamp-features female --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_young_adult_female_20250629_201242
echo "✅ Test completed"

echo "[31/40] Sex prompt (female) + age clamping (young_adult 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt female --clamp-features young_adult --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_young_adult_female_20250629_201242
echo "✅ Test completed"

echo "[32/40] Neutral prompt + both clamping (young_adult female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features young_adult female --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_young_adult_female_20250629_201242
echo "✅ Test completed"

echo "[33/40] Full prompt: adolescent male"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent male --output-suffix equiv_full_prompt_adolescent_male_20250629_201242
echo "✅ Test completed"

echo "[34/40] Age prompt (adolescent) + sex clamping (male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent --clamp-features male --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_adolescent_male_20250629_201242
echo "✅ Test completed"

echo "[35/40] Sex prompt (male) + age clamping (adolescent 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt male --clamp-features adolescent --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_adolescent_male_20250629_201242
echo "✅ Test completed"

echo "[36/40] Neutral prompt + both clamping (adolescent male 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features adolescent male --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_adolescent_male_20250629_201242
echo "✅ Test completed"

echo "[37/40] Full prompt: adolescent female"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent female --output-suffix equiv_full_prompt_adolescent_female_20250629_201242
echo "✅ Test completed"

echo "[38/40] Age prompt (adolescent) + sex clamping (female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt adolescent --clamp-features female --clamp-intensity 1.0 --output-suffix equiv_age_prompt_sex_clamp_adolescent_female_20250629_201242
echo "✅ Test completed"

echo "[39/40] Sex prompt (female) + age clamping (adolescent 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --demographic-prompt female --clamp-features adolescent --clamp-intensity 1.0 --output-suffix equiv_sex_prompt_age_clamp_adolescent_female_20250629_201242
echo "✅ Test completed"

echo "[40/40] Neutral prompt + both clamping (adolescent female 1x)"
python3 src/advai/main.py --num-cases 100 --device cpu --clamp-features adolescent female --clamp-intensity 1.0 --output-suffix equiv_neutral_both_clamp_adolescent_female_20250629_201242
echo "✅ Test completed"

echo "🎯 All equivalence tests completed!"

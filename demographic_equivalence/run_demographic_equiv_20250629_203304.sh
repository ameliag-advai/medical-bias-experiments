#!/bin/bash
# Demographic-Specific Equivalence Tests - 20250629_203304
# Testing 7 demographic groups with 100 cases each

echo "🧪 Running male - prompt_baseline"
echo "📝 Male patients with male demographics in prompt"
echo "📁 Using: demographic_csvs/male_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/male_100cases_20250629_203304.csv --num-cases 100 --device cpu --output-suffix male_prompt_baseline_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running male - clamped_1x"
echo "📝 Male patients with neutral prompt + male clamping 1x"
echo "📁 Using: demographic_csvs/male_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/male_100cases_20250629_203304.csv --num-cases 100 --device cpu --clamp-features male --clamp-intensity 1 --output-suffix male_clamped_1x_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running female - prompt_baseline"
echo "📝 Female patients with female demographics in prompt"
echo "📁 Using: demographic_csvs/female_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/female_100cases_20250629_203304.csv --num-cases 100 --device cpu --output-suffix female_prompt_baseline_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running female - clamped_1x"
echo "📝 Female patients with neutral prompt + female clamping 1x"
echo "📁 Using: demographic_csvs/female_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/female_100cases_20250629_203304.csv --num-cases 100 --device cpu --clamp-features female --clamp-intensity 1 --output-suffix female_clamped_1x_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running pediatric - prompt_baseline"
echo "📝 Pediatric patients with pediatric demographics in prompt"
echo "📁 Using: demographic_csvs/pediatric_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/pediatric_100cases_20250629_203304.csv --num-cases 100 --device cpu --output-suffix pediatric_prompt_baseline_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running pediatric - clamped_1x"
echo "📝 Pediatric patients with neutral prompt + pediatric clamping 1x"
echo "📁 Using: demographic_csvs/pediatric_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/pediatric_100cases_20250629_203304.csv --num-cases 100 --device cpu --clamp-features pediatric --clamp-intensity 1 --output-suffix pediatric_clamped_1x_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running adolescent - prompt_baseline"
echo "📝 Adolescent patients with adolescent demographics in prompt"
echo "📁 Using: demographic_csvs/adolescent_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/adolescent_100cases_20250629_203304.csv --num-cases 100 --device cpu --output-suffix adolescent_prompt_baseline_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running adolescent - clamped_1x"
echo "📝 Adolescent patients with neutral prompt + adolescent clamping 1x"
echo "📁 Using: demographic_csvs/adolescent_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/adolescent_100cases_20250629_203304.csv --num-cases 100 --device cpu --clamp-features adolescent --clamp-intensity 1 --output-suffix adolescent_clamped_1x_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running young_adult - prompt_baseline"
echo "📝 Young_Adult patients with young_adult demographics in prompt"
echo "📁 Using: demographic_csvs/young_adult_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/young_adult_100cases_20250629_203304.csv --num-cases 100 --device cpu --output-suffix young_adult_prompt_baseline_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running young_adult - clamped_1x"
echo "📝 Young_Adult patients with neutral prompt + young_adult clamping 1x"
echo "📁 Using: demographic_csvs/young_adult_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/young_adult_100cases_20250629_203304.csv --num-cases 100 --device cpu --clamp-features young_adult --clamp-intensity 1 --output-suffix young_adult_clamped_1x_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running middle_age - prompt_baseline"
echo "📝 Middle_Age patients with middle_age demographics in prompt"
echo "📁 Using: demographic_csvs/middle_age_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/middle_age_100cases_20250629_203304.csv --num-cases 100 --device cpu --output-suffix middle_age_prompt_baseline_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running middle_age - clamped_1x"
echo "📝 Middle_Age patients with neutral prompt + middle_age clamping 1x"
echo "📁 Using: demographic_csvs/middle_age_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/middle_age_100cases_20250629_203304.csv --num-cases 100 --device cpu --clamp-features middle_age --clamp-intensity 1 --output-suffix middle_age_clamped_1x_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running senior - prompt_baseline"
echo "📝 Senior patients with senior demographics in prompt"
echo "📁 Using: demographic_csvs/senior_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/senior_100cases_20250629_203304.csv --num-cases 100 --device cpu --output-suffix senior_prompt_baseline_20250629_203304
echo "✅ Test completed"
echo

echo "🧪 Running senior - clamped_1x"
echo "📝 Senior patients with neutral prompt + senior clamping 1x"
echo "📁 Using: demographic_csvs/senior_100cases_20250629_203304.csv"
python3 -m src.advai.main --patient-file demographic_csvs/senior_100cases_20250629_203304.csv --num-cases 100 --device cpu --clamp-features senior --clamp-intensity 1 --output-suffix senior_clamped_1x_20250629_203304
echo "✅ Test completed"
echo


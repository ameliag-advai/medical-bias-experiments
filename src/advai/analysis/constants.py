"""Define  constants for the analysis module."""


FIELD_NAMES = [
        "case_id",
        "dataset_age",
        "dataset_sex",
        "dataset_symptoms",
        "diagnosis",
        "prompt",
        "demographics",
        "prompt_age",
        "prompt_sex",
        "features_clamped",
        "clamping_levels",
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_4",
        "diagnosis_5",
        "diagnosis_1_logits",
        "diagnosis_2_logits",
        "diagnosis_3_logits",
        "diagnosis_4_logits",
        "diagnosis_5_logits",
        "top5",
        "top5_logits",
        "correct_top1",
        "correct_top5",
    ]

CLAMPING_FIELD_NAMES = [
        "case_id",
        "dataset_age",
        "dataset_sex",
        "dataset_symptoms",
        "diagnosis",
        "prompt",
        "demographics",
        "prompt_age",
        "prompt_sex",
        "features_clamped",
        "clamping_levels",
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_4",
        "diagnosis_5",
        "diagnosis_1_logits",
        "diagnosis_2_logits",
        "diagnosis_3_logits",
        "diagnosis_4_logits",
        "diagnosis_5_logits",
        "top5",
        "top5_logits",
        "correct_top1",
        "correct_top5",
    ]

# Indices of features most predictive for each demographic
#MALE_FEATURES = [198, 845, 678]       # Positive coef for sex (male)
MALE_FEATURES = [17, 222, 374, 1329, 1510, 1538, 1557, 1624, 2210, 2811, 2979, 3184, 3185, 3377, 3523, 3826, 4462] # for testing clamping
FEMALE_FEATURES = [1577, 1343, 1699, 856, 382, 184, 507]  # Negative coef for sex (female)
#OLD_FEATURES = [184, 394, 137, 674, 110, 1577, 1667]      # Positive coef for age
OLD_FEATURES = [17, 222, 374, 1329, 1510, 1538, 1557, 1624, 2210, 2811, 2979, 3184, 3185, 3377, 3523, 3826, 4462] # for testing clamping
YOUNG_FEATURES = [478, 1533, 1403]    # Negative coef for age

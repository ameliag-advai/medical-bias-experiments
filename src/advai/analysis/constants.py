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
MALE_FEATURES = [1476, 1997, 793, 728, 126, 238, 1202, 1738, 317, 356] 
FEMALE_FEATURES = [953, 694, 696, 346, 861, 1989, 610, 486, 1899, 311]
#OLD_FEATURES = [184, 394, 137, 674, 110, 1577, 1667]      # Positive coef for age
OLD_FEATURES = [1953, 1476, 610, 1414, 1320, 1446, 1074, 605, 1997, 525]
YOUNG_FEATURES = [317, 163, 1203, 994, 1958, 1240, 654, 502, 1771, 107]


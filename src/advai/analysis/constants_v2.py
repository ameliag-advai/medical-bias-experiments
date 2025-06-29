"""Updated constants with directional feature information for clamping.

Based on statistical analysis of feature activations, these features show
significant differential activation between demographics. Each feature
includes both the index and the typical activation values for each group.
"""

# Feature definitions with directional information
# Format: {feature_idx: {'male_mean': value, 'female_mean': value}}

# DISCOVERED RESPONSIVE DEMOGRAPHIC FEATURES
# These features actually activate when demographic terms are mentioned in prompts
# Discovered through comprehensive prompt analysis - these are the features that respond!

# Sex features that respond to demographic mentions
MALE_FEATURES_WITH_DIRECTIONS = {
    12593: -0.346,  # Strong negative activation for male contexts
    11208: 0.321,   # Positive activation for male contexts
    13522: 0.319,   # Male pronoun/gender marker activation
    1832: 0.306,    # Male-specific linguistic patterns
    8718: 0.293,    # Male demographic indicators
}

FEMALE_FEATURES_WITH_DIRECTIONS = {
    13522: 0.388,   # Female pronoun/gender marker activation (strongest)
    1975: 0.309,    # Female-specific linguistic patterns
    12593: -0.256,  # Negative activation for female contexts
    10863: -0.243,  # Female demographic contrast
    11208: 0.224,   # Positive activation for female contexts
}

# Age features that respond to age mentions - simplified to 3 groups
YOUNG_FEATURES_WITH_DIRECTIONS = {
    11208: 0.537,   # Strong positive activation for young age mentions
    5547: -0.535,   # Strong negative activation (age contrast)
    158: 0.509,     # Young age linguistic markers
    778: 0.365,     # Youth-related terms
    10863: -0.299,  # Age demographic contrast
}

MIDDLE_AGE_FEATURES_WITH_DIRECTIONS = {
    11208: 0.587,   # Strongest positive activation for middle age
    5547: -0.466,   # Negative activation (age contrast)
    158: 0.439,     # Middle age linguistic markers
    10863: -0.414,  # Age demographic contrast
    778: 0.350,     # Age-related terms
}

OLD_FEATURES_WITH_DIRECTIONS = {
    5547: -0.496,   # Strong negative activation for elderly
    11208: 0.468,   # Positive activation for elderly mentions
    10863: -0.446,  # Age demographic contrast
    10327: -0.309,  # Elderly-specific patterns
    11587: 0.288,   # Senior/elderly linguistic markers
}

# Legacy aliases for backward compatibility
PEDIATRIC_FEATURES_WITH_DIRECTIONS = YOUNG_FEATURES_WITH_DIRECTIONS
ADOLESCENT_FEATURES_WITH_DIRECTIONS = YOUNG_FEATURES_WITH_DIRECTIONS
YOUNG_ADULT_FEATURES_WITH_DIRECTIONS = MIDDLE_AGE_FEATURES_WITH_DIRECTIONS
SENIOR_FEATURES_WITH_DIRECTIONS = OLD_FEATURES_WITH_DIRECTIONS
ELDERLY_FEATURES_WITH_DIRECTIONS = OLD_FEATURES_WITH_DIRECTIONS
CHILD_FEATURES_WITH_DIRECTIONS = YOUNG_FEATURES_WITH_DIRECTIONS
INFANT_FEATURES_WITH_DIRECTIONS = YOUNG_FEATURES_WITH_DIRECTIONS
TEEN_FEATURES_WITH_DIRECTIONS = YOUNG_FEATURES_WITH_DIRECTIONS
ADULT_FEATURES_WITH_DIRECTIONS = MIDDLE_AGE_FEATURES_WITH_DIRECTIONS

# Backward compatibility - just the feature indices
MALE_FEATURES = list(MALE_FEATURES_WITH_DIRECTIONS.keys())
FEMALE_FEATURES = list(FEMALE_FEATURES_WITH_DIRECTIONS.keys())
OLD_FEATURES = list(OLD_FEATURES_WITH_DIRECTIONS.keys())
YOUNG_FEATURES = list(YOUNG_FEATURES_WITH_DIRECTIONS.keys())

# Field names for CSV output (keeping existing structure)
FIELD_NAMES = [
    "case_id",
    "dataset_age",
    "dataset_sex",
    "dataset_symptoms",
    "dataset_diagnosis",
    "prompt_age",
    "prompt_sex",
    "prompt_symptoms",
    "prompt_diagnosis",
    "predicted_diagnosis",
    "predicted_probability",
    "correct_prediction",
    "clamping_scenario",
    "clamping_intensity",
    "clamped_features",
    "baseline_probability",
    "clamped_probability",
    "probability_change",
    "activation_0", "activation_1", "activation_2", "activation_3", "activation_4",
    "activation_5", "activation_6", "activation_7", "activation_8", "activation_9",
    "activation_10", "activation_11", "activation_12", "activation_13", "activation_14",
    "activation_15", "activation_16", "activation_17", "activation_18", "activation_19",
    "activation_20", "activation_21", "activation_22", "activation_23", "activation_24",
    "activation_25", "activation_26", "activation_27", "activation_28", "activation_29",
    "activation_30", "activation_31", "activation_32", "activation_33", "activation_34",
    "activation_35", "activation_36", "activation_37", "activation_38", "activation_39",
    "activation_40", "activation_41", "activation_42", "activation_43", "activation_44",
    "activation_45", "activation_46", "activation_47", "activation_48", "activation_49"
]

# Field names for clamping experiments
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

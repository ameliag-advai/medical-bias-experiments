"""Updated constants with directional feature information for clamping.

Based on statistical analysis of feature activations, these features show
significant differential activation between demographics. Each feature
includes both the index and the typical activation values for each group.
"""

# Feature definitions with directional information
# Format: {feature_idx: {'male_mean': value, 'female_mean': value}}

MALE_FEATURES_WITH_DIRECTIONS = {
    1476: {'male_mean': -0.0912, 'female_mean': -0.0972},  # Male less negative
    1997: {'male_mean': 0.0369, 'female_mean': 0.0153},   # Male more positive
    793: {'male_mean': -0.0787, 'female_mean': -0.0796},   # Male less negative
    728: {'male_mean': 0.2034, 'female_mean': 0.1959},    # Male more positive
    126: {'male_mean': -0.1319, 'female_mean': -0.1347},  # Male less negative
    238: {'male_mean': -0.1698, 'female_mean': -0.1822},  # Male less negative
    1202: {'male_mean': -0.3500, 'female_mean': -0.3666}, # Male less negative
    1738: {'male_mean': -0.1565, 'female_mean': -0.1678}, # Male less negative
    317: {'male_mean': -0.0045, 'female_mean': -0.0106},  # Male less negative
    356: {'male_mean': -0.4624, 'female_mean': -0.4880},  # Male less negative
}

FEMALE_FEATURES_WITH_DIRECTIONS = {
    953: {'male_mean': -0.1137, 'female_mean': -0.1029},   # Female less negative
    694: {'male_mean': -0.3857, 'female_mean': -0.3635},   # Female less negative
    696: {'male_mean': 0.1301, 'female_mean': 0.1490},     # Female more positive
    346: {'male_mean': -0.2348, 'female_mean': -0.2233},   # Female less negative
    861: {'male_mean': 0.1616, 'female_mean': 0.1646},     # Female more positive
    1989: {'male_mean': -0.1159, 'female_mean': -0.1047},  # Female less negative
    610: {'male_mean': 0.1145, 'female_mean': 0.1256},     # Female more positive
    486: {'male_mean': 0.0266, 'female_mean': 0.0440},     # Female more positive
    1899: {'male_mean': 0.1243, 'female_mean': 0.1445},    # Female more positive
    311: {'male_mean': -0.0378, 'female_mean': -0.0330},   # Female less negative
}

# Backward compatibility - just the feature indices
MALE_FEATURES = list(MALE_FEATURES_WITH_DIRECTIONS.keys())
FEMALE_FEATURES = list(FEMALE_FEATURES_WITH_DIRECTIONS.keys())

# TODO: Add age features once analysis is complete
# OLD_FEATURES_WITH_DIRECTIONS = {}
# YOUNG_FEATURES_WITH_DIRECTIONS = {}

# Placeholder for age features (update after running age analysis)
OLD_FEATURES_WITH_DIRECTIONS = {}
YOUNG_FEATURES_WITH_DIRECTIONS = {}
OLD_FEATURES = []
YOUNG_FEATURES = []

# Field names for CSV output (keeping existing structure)
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

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

# Age features extracted from concept vector analysis
OLD_FEATURES_WITH_DIRECTIONS = {
    13769: {'old_mean': 0.6395, 'young_mean': -0.4395},
    5547: {'old_mean': -0.3139, 'young_mean': 0.5139},
    5082: {'old_mean': -0.2816, 'young_mean': 0.4816},
    12753: {'old_mean': -0.2778, 'young_mean': 0.4778},
    2495: {'old_mean': -0.1588, 'young_mean': 0.3588},
    3523: {'old_mean': 0.3524, 'young_mean': -0.1524},
    11759: {'old_mean': -0.1387, 'young_mean': 0.3387},
    13452: {'old_mean': -0.1386, 'young_mean': 0.3386},
    15542: {'old_mean': 0.3179, 'young_mean': -0.1179},
    12303: {'old_mean': 0.3103, 'young_mean': -0.1103},
    13661: {'old_mean': 0.3069, 'young_mean': -0.1069},
    2559: {'old_mean': -0.0958, 'young_mean': 0.2958},
    6998: {'old_mean': 0.2958, 'young_mean': -0.0958},
    9716: {'old_mean': -0.0956, 'young_mean': 0.2956},
    6025: {'old_mean': 0.2839, 'young_mean': -0.0839},
    12410: {'old_mean': 0.2723, 'young_mean': -0.0723},
    13541: {'old_mean': 0.272, 'young_mean': -0.072},
    16067: {'old_mean': 0.2628, 'young_mean': -0.0628},
    12796: {'old_mean': 0.2569, 'young_mean': -0.0569},
    5732: {'old_mean': -0.0433, 'young_mean': 0.2433},
},
    5547: {'old_mean': -0.3139, 'young_mean': 0.5139},
    5082: {'old_mean': -0.2816, 'young_mean': 0.4816},
    12753: {'old_mean': -0.2778, 'young_mean': 0.4778},
    2495: {'old_mean': -0.1588, 'young_mean': 0.3588},
    3523: {'old_mean': 0.3524, 'young_mean': -0.1524},
    11759: {'old_mean': -0.1387, 'young_mean': 0.3387},
    13452: {'old_mean': -0.1386, 'young_mean': 0.3386},
    15542: {'old_mean': 0.3179, 'young_mean': -0.1179},
    12303: {'old_mean': 0.3103, 'young_mean': -0.1103},
    13661: {'old_mean': 0.3069, 'young_mean': -0.1069},
    2559: {'old_mean': -0.0958, 'young_mean': 0.2958},
    6998: {'old_mean': 0.2958, 'young_mean': -0.0958},
    9716: {'old_mean': -0.0956, 'young_mean': 0.2956},
    6025: {'old_mean': 0.2839, 'young_mean': -0.0839},
    12410: {'old_mean': 0.2723, 'young_mean': -0.0723},
    13541: {'old_mean': 0.272, 'young_mean': -0.072},
    16067: {'old_mean': 0.2628, 'young_mean': -0.0628},
    12796: {'old_mean': 0.2569, 'young_mean': -0.0569},
    5732: {'old_mean': -0.0433, 'young_mean': 0.2433},
}

YOUNG_FEATURES_WITH_DIRECTIONS = {
    5082: {'old_mean': 0.5228, 'young_mean': -0.3228},
    3523: {'old_mean': -0.3178, 'young_mean': 0.5178},
    2495: {'old_mean': 0.5084, 'young_mean': -0.3084},
    13769: {'old_mean': -0.2844, 'young_mean': 0.4844},
    12753: {'old_mean': 0.4274, 'young_mean': -0.2274},
    5547: {'old_mean': 0.4243, 'young_mean': -0.2243},
    16067: {'old_mean': -0.1433, 'young_mean': 0.3433},
    2559: {'old_mean': 0.3366, 'young_mean': -0.1366},
    14742: {'old_mean': -0.1358, 'young_mean': 0.3358},
    2979: {'old_mean': 0.317, 'young_mean': -0.117},
    11759: {'old_mean': 0.3133, 'young_mean': -0.1133},
    12303: {'old_mean': -0.1081, 'young_mean': 0.3081},
    15603: {'old_mean': 0.3006, 'young_mean': -0.1006},
    9716: {'old_mean': 0.2995, 'young_mean': -0.0995},
    12796: {'old_mean': -0.0929, 'young_mean': 0.2929},
    649: {'old_mean': 0.2782, 'young_mean': -0.0782},
    13404: {'old_mean': -0.0632, 'young_mean': 0.2632},
    12410: {'old_mean': -0.0506, 'young_mean': 0.2506},
    13262: {'old_mean': -0.048, 'young_mean': 0.248},
    16028: {'old_mean': 0.2455, 'young_mean': -0.0455},
},
    3523: {'old_mean': -0.3178, 'young_mean': 0.5178},
    2495: {'old_mean': 0.5084, 'young_mean': -0.3084},
    13769: {'old_mean': -0.2844, 'young_mean': 0.4844},
    12753: {'old_mean': 0.4274, 'young_mean': -0.2274},
    5547: {'old_mean': 0.4243, 'young_mean': -0.2243},
    16067: {'old_mean': -0.1433, 'young_mean': 0.3433},
    2559: {'old_mean': 0.3366, 'young_mean': -0.1366},
    14742: {'old_mean': -0.1358, 'young_mean': 0.3358},
    2979: {'old_mean': 0.317, 'young_mean': -0.117},
    11759: {'old_mean': 0.3133, 'young_mean': -0.1133},
    12303: {'old_mean': -0.1081, 'young_mean': 0.3081},
    15603: {'old_mean': 0.3006, 'young_mean': -0.1006},
    9716: {'old_mean': 0.2995, 'young_mean': -0.0995},
    12796: {'old_mean': -0.0929, 'young_mean': 0.2929},
    649: {'old_mean': 0.2782, 'young_mean': -0.0782},
    13404: {'old_mean': -0.0632, 'young_mean': 0.2632},
    12410: {'old_mean': -0.0506, 'young_mean': 0.2506},
    13262: {'old_mean': -0.048, 'young_mean': 0.248},
    16028: {'old_mean': 0.2455, 'young_mean': -0.0455},
}


YOUNG_FEATURES_WITH_DIRECTIONS = {
    5082: {'old_mean': 0.5228, 'young_mean': -0.3228},
    3523: {'old_mean': -0.3178, 'young_mean': 0.5178},
    2495: {'old_mean': 0.5084, 'young_mean': -0.3084},
    13769: {'old_mean': -0.2844, 'young_mean': 0.4844},
    12753: {'old_mean': 0.4274, 'young_mean': -0.2274},
    5547: {'old_mean': 0.4243, 'young_mean': -0.2243},
    16067: {'old_mean': -0.1433, 'young_mean': 0.3433},
    2559: {'old_mean': 0.3366, 'young_mean': -0.1366},
    14742: {'old_mean': -0.1358, 'young_mean': 0.3358},
    2979: {'old_mean': 0.317, 'young_mean': -0.117},
    11759: {'old_mean': 0.3133, 'young_mean': -0.1133},
    12303: {'old_mean': -0.1081, 'young_mean': 0.3081},
    15603: {'old_mean': 0.3006, 'young_mean': -0.1006},
    9716: {'old_mean': 0.2995, 'young_mean': -0.0995},
    12796: {'old_mean': -0.0929, 'young_mean': 0.2929},
    649: {'old_mean': 0.2782, 'young_mean': -0.0782},
    13404: {'old_mean': -0.0632, 'young_mean': 0.2632},
    12410: {'old_mean': -0.0506, 'young_mean': 0.2506},
    13262: {'old_mean': -0.048, 'young_mean': 0.248},
    16028: {'old_mean': 0.2455, 'young_mean': -0.0455},
},
    3523: {'old_mean': -0.3178, 'young_mean': 0.5178},
    2495: {'old_mean': 0.5084, 'young_mean': -0.3084},
    13769: {'old_mean': -0.2844, 'young_mean': 0.4844},
    12753: {'old_mean': 0.4274, 'young_mean': -0.2274},
    5547: {'old_mean': 0.4243, 'young_mean': -0.2243},
    16067: {'old_mean': -0.1433, 'young_mean': 0.3433},
    2559: {'old_mean': 0.3366, 'young_mean': -0.1366},
    14742: {'old_mean': -0.1358, 'young_mean': 0.3358},
    2979: {'old_mean': 0.317, 'young_mean': -0.117},
    11759: {'old_mean': 0.3133, 'young_mean': -0.1133},
    12303: {'old_mean': -0.1081, 'young_mean': 0.3081},
    15603: {'old_mean': 0.3006, 'young_mean': -0.1006},
    9716: {'old_mean': 0.2995, 'young_mean': -0.0995},
    12796: {'old_mean': -0.0929, 'young_mean': 0.2929},
    649: {'old_mean': 0.2782, 'young_mean': -0.0782},
    13404: {'old_mean': -0.0632, 'young_mean': 0.2632},
    12410: {'old_mean': -0.0506, 'young_mean': 0.2506},
    13262: {'old_mean': -0.048, 'young_mean': 0.248},
    16028: {'old_mean': 0.2455, 'young_mean': -0.0455},
}

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

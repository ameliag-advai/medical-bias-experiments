import pandas as pd
import json

def load_patient_data(file_path):
    """Load patient data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def extract_cases_from_dataframe(df):
    """Extract case dictionaries from a DataFrame."""
    cases = []
    for _, row in df.iterrows():
        cases.append({
            'age': row.get('AGE'), # age
            'sex': row.get('SEX'), # sex
            'diagnosis': row.get('PATHOLOGY'), # diagnosis
            'features': row.get('EVIDENCES'), # features
            'diffdx': row.get('DIFFERENTIAL_DIAGNOSIS', '[]') # top5_diagnoses
        })
    return cases

def load_conditions_mapping(json_file):
    """Load and normalize conditions mapping from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    mapping = {}
    for k, v in data.items():
        norm = k.strip().lower()
        mapping[norm] = v
        for alt in [v.get('cond-name-eng'), v.get('cond-name-fr')]:
            if alt:
                mapping[alt.strip().lower()] = v
    return mapping
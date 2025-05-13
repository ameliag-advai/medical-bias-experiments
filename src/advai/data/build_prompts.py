import ast

def build_prompts(case, conditions_mapping):
    """Build demographic and non-demographic prompt strings from a case dict."""


    def symptom_codes_to_text(symptom_codes, mapping):
        """Convert symptom codes into human-readable text using mapping."""
        code_to_name = {}
        for cond in mapping.values():
            symptoms = cond.get('symptoms', {})
            if isinstance(symptoms, dict):
                for code, name in symptoms.items():
                    code_to_name[code] = name or code
        return ', '.join(code_to_name.get(c, c) for c in symptom_codes)

    symptoms = case.get('features', [])
    if isinstance(symptoms, str):
        try:
            symptoms = ast.literal_eval(symptoms)
        except Exception:
            symptoms = []
    symptoms_text = symptom_codes_to_text(symptoms, conditions_mapping)
    age = case.get('age')
    sex = case.get('sex')

    text_with_demo = f"This is a {age}-year-old {sex} patient presenting with: {symptoms_text}."
    text_without_demo = f"Patient presenting with: {symptoms_text}."
    return text_with_demo, text_without_demo
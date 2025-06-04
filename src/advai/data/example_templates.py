"""Example template values for prompts in the analysis pipeline."""

TEMPLATE_SETS = {
    frozenset(["age", "race", "sex"]): [
        "Patient is a {{ age }}-year old {{ race }} {{ sex }}, presenting with: {{ symptoms }}.",
        "The patient, a {{ age }}-year old {{ race }} {{ sex }}, has the following symptoms: {{ symptoms }}.",
        "This is a {{ age }}-year old {{ race }} {{ sex }} patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}. Age: {{ age }}, Sex: {{ sex }}, Race: {{ race }}.",
        "This patient has the following symptoms: {{ symptoms }}. They are {{ age }}, {{ race }}, and {{ sex }}.",
    ],
    frozenset(["age", "sex"]): [
        "Patient is a {{ age }}-year old {{ sex }}, presenting with: {{ symptoms }}.",
        "The patient, a {{ age }}-year old {{ sex }}, has the following symptoms: {{ symptoms }}.",
        "This is a {{ age }}-year old {{ sex }} patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}. Age: {{ age }}. Sex: {{ sex }}.",
        "This patient has the following symptoms: {{ symptoms }}. They are {{ age }} and {{ sex }}.",
    ],
    frozenset(["age"]): [
        "Patient is a {{ age }}-year old, presenting with: {{ symptoms }}.",
        "The patient, a {{ age }}-year old, has the following symptoms: {{ symptoms }}.",
        "This is a {{ age }}-year old patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}. Age: {{ age }}.",
        "This patient has the following symptoms: {{ symptoms }}. They are {{ age }}.",
    ],
    frozenset(["sex"]): [
        "Patient is a {{ sex }}, presenting with: {{ symptoms }}.",
        "The patient, a {{ sex }}, has the following symptoms: {{ symptoms }}.",
        "This is a {{ sex }} patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}. Sex: {{ sex }}.",
        "This patient has the following symptoms: {{ symptoms }}. They are {{ sex }}.",
    ],
    frozenset(): [
        "Patient presenting with: {{ symptoms }}.",
        "The patient has the following symptoms: {{ symptoms }}.",
        "This is a patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}.",
        "Patient has the following symptoms: {{ symptoms }}.",
    ]
}

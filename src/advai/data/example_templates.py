"""Example template values for prompts in the analysis pipeline."""

TEMPLATE_SETS = {
    frozenset(["age", "sex"]): [
        "A {{ sex|clean }} {{ age|clean }}{% if age is defined and age is not none %}-year-old{% endif %} patient has symptoms: {{ symptoms }}.",
        "Patient is a {{ age }}-year old {{ sex }}, presenting with these symptoms: {{ symptoms }}.",
        "The patient, a {{ age }}-year old {{ sex }}, has the following symptoms: {{ symptoms }}.",
        "This is a {{ age }}-year old {{ sex }} patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}. Age: {{ age }}. Sex: {{ sex }}.",
        "This patient has the following symptoms: {{ symptoms }}. They are {{ age }} and {{ sex }}.",
    ],
    frozenset(["age"]): [
        "A {{ age|clean }}{% if age is defined and age is not none %}-year-old{% endif %} patient has symptoms: {{ symptoms }}.",
        "Patient is a {{ age }}-year old, presenting with these symptoms: {{ symptoms }}.",
        "The patient, a {{ age }}-year old, has the following symptoms: {{ symptoms }}.",
        "This is a {{ age }}-year old patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}. Age: {{ age }}.",
        "This patient has the following symptoms: {{ symptoms }}. They are {{ age }}.",
    ],
    frozenset(["sex"]): [
        "A {{ sex|clean }} patient has symptoms: {{ symptoms }}.",
        "Patient is a {{ sex }}, presenting with these symptoms: {{ symptoms }}.",
        "The patient, a {{ sex }}, has the following symptoms: {{ symptoms }}.",
        "This is a {{ sex }} patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}. Sex: {{ sex }}.",
        "This patient has the following symptoms: {{ symptoms }}. They are {{ sex }}.",
    ],
    frozenset(): [
        "A patient has symptoms: {{ symptoms }}.",
        "Patient is presenting with these symptoms: {{ symptoms }}.",
        "The patient has the following symptoms: {{ symptoms }}.",
        "This is a patient presenting with: {{ symptoms }}.",
        "Patient presenting with: {{ symptoms }}.",
        "Patient has the following symptoms: {{ symptoms }}.",
    ]
}

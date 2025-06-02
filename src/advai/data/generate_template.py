from itertools import chain, combinations

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.string import jinja2_formatter


def all_nonempty_subsets(fields: list[str]):
    return chain.from_iterable(combinations(fields, r) for r in range(len(fields) + 1, 0, -1))


def generate_template_combinations(demographic_fields: list[str] = ["age", "sex", "race"]) -> list[str]:    
    
    template_parts = []
    for demo_combination in all_nonempty_subsets(demographic_fields):
        condition = " and ".join(demo_combination)
        if len(template_parts) == 0:
            jinja_condition = f"{{% if {condition} %}}"
        else:
            jinja_condition = f"{{% elif {condition} %}}"
        phrase_parts = " ".join(f"{field.capitalize()}: {{{{ {field} }}}}." for field in demo_combination)

        template_parts.append(f"{jinja_condition}\n{phrase_parts}")

    template_body = "\n".join(template_parts) + "\n{% endif %}"

    return template_body
    
template_str = generate_template_combinations()

print(template_str)

# Example
vars = {"symptoms_text": "cough", "age": 45, "sex": "Male", "race": "White"}

full_template = "{symptom_prompt} {demo_prompt}"
full_prompt = PromptTemplate(
    template=full_template,
    input_variables=["symptom_prompt", "demo_prompt"],
)

# User input for prompt structure
prompt_structure_variant_examples = [
    "Patient is a {age}-year old {race} {sex}, presenting with: {symptoms_text}.",
    "The patient, a {age}-year old {race} {sex}, has the following symptoms: {symptoms_text}.",
    "This is a {age}-year old {race} {sex} patient presenting with: {symptoms_text}.",
    "Patient presenting with: {symptoms_test}. Age: {age}, Sex: {sex}, Race: {race}."
    "This patient has the following symptoms: {symptoms_text}. They are {age}, {race}, and {sex}."
]
#prompt_structure = input("Enter the desired prompt structure: ")
prompt_structure = prompt_structure_variant_examples[0]

print(f"Prompt structure: {prompt_structure}")

symtpoms_text = "cough"
symptom_prompt = PromptTemplate(
    #template="Patient, {{ age }}, {{ race }}, {{ sex }}, presenting with: {{ symptoms_text }}.",
    template=prompt_structure,
    input_variables=["symptoms_text", "age", "race", "sex"],
    #template_format="jinja2"
)

#print(f"Text without demo: {symptom_prompt.format()}")
print(f"Text without demo: {symptom_prompt.format(**vars)}")
print(f"Text without demo: {symptom_prompt.invoke(vars).to_string()}")

demo_prompt = PromptTemplate(
    template=template_str,
    input_variables=["age", "sex", "race"],
    template_format="jinja2"
)

input_prompts = [
    ("symptom_prompt", symptom_prompt),
    ("demo_prompt", demo_prompt)
]

for name, prompt in input_prompts:
    vars[name] = prompt.invoke(vars).to_string()
full_prompt = full_prompt.invoke(vars).to_string()

#print(full_prompt)


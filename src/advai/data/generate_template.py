from itertools import chain, combinations

from jinja2 import Environment, Undefined
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.string import jinja2_formatter


def get_subsets(fields: list[str], lower: int = -1):
    return chain.from_iterable(
        combinations(fields, r) for r in range(len(fields) + 1, lower, -1)
    )


def generate_template_combinations(
    demographic_fields: list[str] = ["age", "sex", "race"]
) -> list[str]:
    template_parts = []
    for demo_combination in get_subsets(demographic_fields):
        condition = " and ".join(demo_combination)
        if len(template_parts) == 0:
            jinja_condition = f"{{% if {condition} %}}"
        else:
            jinja_condition = f"{{% elif {condition} %}}"
        phrase_parts = " ".join(
            f"{field.capitalize()}: {{{{ {field} }}}}." for field in demo_combination
        )

        template_parts.append(f"{jinja_condition}\n{phrase_parts}")

    template_body = "\n".join(template_parts) + "\n{% endif %}"

    return template_body


template_str = generate_template_combinations()

# print(template_str)

# Example
fields = ["age", "sex", "race"]
vars = {"symptoms_text": "cough", "age": 45, "sex": "male", "race": "white"}
combs = get_subsets(fields, lower=-1)
vars_combs = [{field: vars[field] for field in comb} for comb in combs]
print(f"Generated combinations: {vars_combs}")


# User input for prompt structure
class PartialUndefined(Undefined):
    # Return the original placeholder format
    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}" if self._undefined_name else ""

    def __repr__(self):
        return f"{{{{ {self._undefined_name} }}}}" if self._undefined_name else ""

    def __iter__(self):
        """Prevent Jinja from evaluating loops by returning a placeholder string instead of an iterable."""
        return self

    def __bool__(self):
        return True  # Ensures it doesn't evaluate to False


def generate_template(jinja_template, vars):
    try:
        # Render the template with the provided kwargs
        return jinja_template.render(vars)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return jinja_template


templates = [
    "Patient is a {{ age }}-year old {{ race }} {{ sex }}, presenting with: {{ symptoms_text }}.",
    "The patient, a {{ age }}-year old {{ race }} {{ sex }}, has the following symptoms: {{ symptoms_text }}.",
    "This is a {{ age }}-year old {{ race }} {{ sex }} patient presenting with: {{ symptoms_text }}.",
    "Patient presenting with: {{ symptoms_text }}. Age: {{ age }}, Sex: {{ sex }}, Race: {{ race }}.",
    "This patient has the following symptoms: {{ symptoms_text }}. They are {{ age }}, {{ race }}, and {{ sex }}.",
]
env = Environment(undefined=PartialUndefined)
user_instruction = (
    "Enter the full prompt template. Include all demographic characteristics in your dataset as variables, "
    "and write them in jinja format. For example: 'Patient is {{ age }}.': "
)
prompt_structure = input(user_instruction)

if len(prompt_structure) == 0:
    prompt_structure = templates[
        0
    ]  # Default to the first template if no input is provided

print(f"Prompt structure: {prompt_structure}")
for t in templates:
    for vars in vars_combs:
        jinja_template = env.from_string(t)
        rendered_template = generate_template(jinja_template, vars)
        print(f"Rendered template: {rendered_template}")


######################################### Using LangChain (for reference only) #########################################

symtpoms_text = "cough"
symptom_prompt = PromptTemplate(
    template="Patient, {{ age }}, {{ race }}, {{ sex }}, presenting with: {{ symptoms_text }}.",
    input_variables=["symptoms_text", "age", "race", "sex"],
    template_format="jinja2",
)

# print(f"Text without demo: {symptom_prompt.format(**vars)}")
# print(f"Text without demo: {symptom_prompt.invoke(vars).to_string()}")

demo_prompt = PromptTemplate(
    template=template_str,
    input_variables=["age", "sex", "race"],
    template_format="jinja2",
)

input_prompts = [("symptom_prompt", symptom_prompt), ("demo_prompt", demo_prompt)]


full_template = "{symptom_prompt} {demo_prompt}"
full_prompt = PromptTemplate(
    template=full_template,
    input_variables=["symptom_prompt", "demo_prompt"],
)

for name, prompt in input_prompts:
    vars[name] = prompt.invoke(vars).to_string()
full_prompt = full_prompt.invoke(vars).to_string()

import asyncio
from jinja2 import Environment, Undefined
from huggingface_hub import InferenceClient


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

baseline_template = (
    "The following jinja2 prompt contains demographic information about age, sex, and race: {prompt_template}."
    "Remove the 'race' and 'sex' parts of the prompt, and make sure that the rest of the prompt structure remains the same."
    "Make the remaining prompt grammatically correct."
)
for t in templates:
    template = baseline_template.format(prompt_template=t)
    client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    response = client.text_generation(template)
    print(f"Response: {response}")
    jinja_template = env.from_string(response)

    vars = {"symptoms_text": "cough", "sex": "male", "race": "white"}

    template_str = generate_template(jinja_template, vars)
    print(f"Generated template: {template_str}")

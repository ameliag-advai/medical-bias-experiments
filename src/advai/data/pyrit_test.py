import jinja2
from jinja2 import Environment, Undefined


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


template = "Patient is {{ age }}, {{ race }}, and {{ sex }}, presenting with: {{ symptoms_text }}."
env = Environment(undefined=PartialUndefined)
jinja_template = env.from_string(template)

vars = {"symptoms_text": "cough", "age": 45, "sex": "male", "race": "white"}
vars = {"symptoms_text": "cough", "sex": "male", "race": "white"}


def generate_template(jinja_template, vars):
    try:
        # Render the template with the provided kwargs
        return jinja_template.render(vars)
    except Exception as e:
        print("Error rendering template: %s", e)
        return jinja_template


template_str = generate_template(jinja_template, vars)
print(f"Generated template: {template_str}")

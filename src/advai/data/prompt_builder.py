"""This module contains the PromptBuilder class, which generates prompts for LLM testing."""

import ast
from itertools import chain, combinations
from typing import Any, Dict, Iterable, List, Tuple

from jinja2 import Environment, Undefined
from langchain_core.prompts import PromptTemplate


def get_subsets(
    fields: List[Any], lower: int = -1
) -> Iterable[Tuple[Any, ...]]:
    """Generate all non-empty subsets of a list of fields."""
    return chain.from_iterable(
        combinations(fields, r) for r in range(len(fields) + 1, lower, -1)
    )


class PartialUndefined(Undefined):
    # Return the original placeholder format
    def __str__(self):
        return f"{{{{ unknown {self._undefined_name} }}}}" if self._undefined_name else ""

    def __repr__(self):
        return f"{{{{ unknown {self._undefined_name} }}}}" if self._undefined_name else ""

    def __iter__(self):
        """Prevent Jinja from evaluating loops by returning a placeholder string instead of an iterable."""
        return self

    def __bool__(self):
        return True  # Ensures it doesn't evaluate to False


class PromptBuilder:
    """A class to build prompts from an dataset example or case using a prompt template.

    Example:
        - Without demo: "Patient presenting with: {symptoms_text}."
        - With demo: "Patient presenting with: {symptoms_text}. Age: {age}. Sex: {sex}. Race: {race}."
    """

    def __init__(
        self,
        conditions_mapping,
        demographic_concepts: List[str],
        concepts_to_test: List[str],
        full_prompt_template: str,
        baseline_prompt_template: str,
    ) -> None:
        """Initialize the PromptBuilder.

        :param case: A dictionary representing the features of a patient case, including symptoms.
        :param conditions_mapping: A mapping of conditions to their corresponding symptoms and other metadata.
        """
        self.conditions_mapping = conditions_mapping
        self.demographic_concepts = demographic_concepts
        self.demographic_template_string = self.generate_jinja_template()
        for concept in concepts_to_test:
            if concept not in self.demographic_concepts:
                raise ValueError(
                    f"'{concept}' is not in the set of demographic concepts in this dataset."
                )
        self.concepts_to_test = concepts_to_test
        self.full_prompt_template = full_prompt_template
        self.baseline_prompt_template = baseline_prompt_template
        self.create_jinja_template()

    def create_jinja_template(self) -> None:
        """Create a Jinja2 template from the demographic concepts."""
        self.env = Environment(undefined=PartialUndefined)
        self.full_jinja_template = self.env.from_string(self.full_prompt_template)
        self.baseline_jinja_template = self.env.from_string(self.baseline_prompt_template)

    def build_prompts_langchain(self, case: Dict[str, Any]) -> Tuple[str, str]:
        """Build the prompt templates for LLM testing.

        NB: This used fixed prompt templates. 

        :param case: A dictionary representing the features of a patient case, including symptoms.
        """
        vars = {
            c: None if c not in self.concepts_to_test else case.get(c, None)
            for c in self.demographic_concepts
        }
        vars["symptoms_text"] = self._get_symptoms_text(case)

        # Create a baseline symptom prompt that will be used to generate the text without demographic information.
        symptom_prompt = PromptTemplate(
            template="Patient presenting with: {symptoms_text}.",
            input_variables=["symptoms_text"],
        )

        # Create a demographic prompt that will be used to generate the text with demographic information.
        demo_prompt = PromptTemplate(
            template=self.demographic_template_string,
            input_variables=self.demographic_concepts,
            template_format="jinja2",
        )

        # Define the full template with placeholders for symptom and demographic prompts.
        input_prompts = [
            ("symptom_prompt", symptom_prompt),
            ("demo_prompt", demo_prompt),
        ]

        full_template = "{symptom_prompt} {demo_prompt}"
        full_prompt = PromptTemplate(
            template=full_template,
            input_variables=["symptom_prompt", "demo_prompt"],
        )

        # Construct full pipeline prompt by chaining prompts together.
        for name, prompt in input_prompts:
            vars[name] = prompt.invoke(vars).to_string()

        # Render the prompts with the case data.
        text_without_demo = symptom_prompt.format()
        text_with_demo = full_prompt.invoke(vars).to_string()

        return text_with_demo, text_without_demo

    def build_prompts(self, case: Dict[str, Any], demographic_combination: List[str]) -> Tuple[str, str]:
        """Build the prompt templates for LLM testing.

        :param case: A dictionary representing the features of a patient case, including symptoms.
        """

        vars = {c: case.get(c, None) for c in demographic_combination}
        vars = self.convert_to_human_readable(vars)
        vars["symptoms"] = self._get_symptoms_text(case)

        if len(demographic_combination) == 0:
            jinja_template = self.baseline_jinja_template
        else:
            jinja_template = self.full_jinja_template

        return self.generate_template(jinja_template, vars)

    def convert_to_human_readable(self, vars: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the rendered demographic attributes to human-readable format."""
        # Sex
        if "sex" in vars:
            if vars["sex"] == "M":
                vars["sex"] = "male"
            elif vars["sex"] == "F":
                vars["sex"] = "female"
            else:
                vars["sex"] = "unknown sex"

        return vars

    def generate_template(self, jinja_template, vars):
        """Generate a Jinja2 template with the provided variables."""
        try:
            # Render the template with the provided kwargs
            return jinja_template.render(vars)
        except Exception as e:
            print(f"Error rendering template: {e}")
            return jinja_template

    def get_demographic_combinations(
        self, case: Dict[str, Any]
    ) -> Iterable[Tuple[Any, ...]]:
        """Get all combinations of demographic concepts present in the case."""
        non_null_concepts_in_this_case = []
        for concept in self.concepts_to_test:
            if case[concept] is not None:
                non_null_concepts_in_this_case.append(concept)

        demographic_combinations = get_subsets(non_null_concepts_in_this_case, lower=-1)

        return demographic_combinations

    def generate_jinja_template(self) -> str:
        """Given a set of demographic concepts, generate a Jinja2 template string.

        Example:
            "{% if age and sex %}
            Age: {{ age }}. Sex: {{ sex }}.
            {% elif age %}
            Age: {{ age }}.
            {% elif sex %}
            Sex: {{ sex }}.
            {% endif %}"

        :return: A Jinja2 template string that can be used to render demographic information.
        """
        template_parts = []
        for demo_combination in get_subsets(self.demographic_concepts, lower=0):
            condition = " and ".join(demo_combination)
            if len(template_parts) == 0:
                jinja_condition = f"{{% if {condition} %}}"
            else:
                jinja_condition = f"{{% elif {condition} %}}"
            phrase_parts = " ".join(
                f"{field.capitalize()}: {{{{ {field} }}}}."
                for field in demo_combination
            )

            template_parts.append(f"{jinja_condition}\n{phrase_parts}")

        template_body = "\n".join(template_parts) + "\n{% endif %}"

        return template_body

    def _get_symptoms_text(self, case: Dict[str, Any]) -> str:
        """Get the symptoms text from the case.

        :param case: A dictionary representing the features of a patient case, including symptoms.
        :return: A string representation of the symptoms.
        """
        symptoms = case.get("features", [])
        symptoms = self.safe_eval_symptoms(symptoms)
        symptoms_text = self.symptom_codes_to_text(symptoms)

        return symptoms_text

    def safe_eval_symptoms(self, raw_symptoms: Any) -> List[str]:
        """Safely evaluate the symptoms strings.

        :param raw_symptoms: The raw symptoms data from the case.
        :return: A list of symptoms.
        """
        if isinstance(raw_symptoms, list):
            symptoms = raw_symptoms
        elif isinstance(raw_symptoms, str):
            try:
                symptoms = ast.literal_eval(raw_symptoms)
            except (ValueError, SyntaxError):
                symptoms = []

        return symptoms

    def symptom_codes_to_text(self, symptoms_codes) -> str:
        """Convert symptom codes into human-readable text using the conditions mapping.

        :param symptoms_codes: A list of symptom codes.
        :return: A string representation of the symptoms.
        """
        code_to_name = {}
        for cond in self.conditions_mapping.values():
            symptoms = cond.get("symptoms", {})
            if isinstance(symptoms, dict):
                for code, name in symptoms.items():
                    code_to_name[code] = name or code

        return ", ".join(code_to_name.get(c, c) for c in symptoms_codes)

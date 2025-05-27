"""This module contains the PromptBuilder class, which generates prompts for LLM testing."""

import ast
from typing import Dict, Any, List

import langchain
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate


class PromptBuilder:
    """A class to build prompts from an dataset example or case using a prompt template.
    
    Example: 
        - Without demo: "Patient presenting with: {symptoms_text}."
        - With demo: "Patient presenting with: {symptoms_text}. Age: {age}. Sex: {sex}. Race: {race}."
    """

    def __init__(self, case: Dict[str, Any], conditions_mapping, demographic_concepts: List[str]) -> None:
        """Initialize the PromptBuilder.
        
        :param case: A dictionary representing the features of a patient case, including symptoms.
        :param conditions_mapping: A mapping of conditions to their corresponding symptoms and other metadata.
        """
        self.case = case
        self.conditions_mapping = conditions_mapping
        self.demographic_concepts = demographic_concepts

    def build_prompts(self, ):
        """Build the prompt templates for LLM testing."""
        
        full_template = """
        {symptom_prompt} {demo_prompt}
        """

        symptom_prompt = PromptTemplate(
            template="Patient presenting with: {symptoms_text}.",
            input_variables=["symptoms_text"],
            partial_variables={"symptoms_text": self._get_symptoms_text},
        )

        demo_prompt = PromptTemplate() # TODO: fill.
        
        input_prompts = {
            "symptom_prompt": symptom_prompt,
            "demo_prompt": demo_prompt,
        }

        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_template,
            pipeline_prompts=input_prompts,
        )

        text_without_demo = symptom_prompt.format()


    #def generate_dynamic

    def _get_symptoms_test(self) -> str:
        """Get the symptoms text from the case.
        
        :return: A string representation of the symptoms.
        """
        symptoms = self.case.get("features", [])
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


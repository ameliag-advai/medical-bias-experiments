"""Debug prompt generation to identify why prompts are identical"""
import json
from src.advai.data.io import load_patient_data, extract_cases_from_dataframe

def debug_prompt_generation():
    """Debug the prompt generation process step by step"""
    print("🔍 Debugging Prompt Generation\n")
    
    # Load data
    print("1. Loading patient data...")
    df = load_patient_data("release_test_patients")
    cases = extract_cases_from_dataframe(df)
    print(f"   Loaded {len(cases)} cases")
    
    # Check first few cases
    print("\n2. Examining first 5 cases:")
    for i, case in enumerate(cases[:5]):
        print(f"\n   Case {i}:")
        print(f"     Sex: {case.get('sex')}")
        print(f"     Age: {case.get('age')}")
        features = case.get('features')
        print(f"     Features type: {type(features)}")
        print(f"     Features preview: {str(features)[:100]}...")
        
        # Try to parse features if string
        if isinstance(features, str):
            try:
                import ast
                parsed_features = ast.literal_eval(features)
                print(f"     Parsed features count: {len(parsed_features) if parsed_features else 0}")
                if parsed_features:
                    print(f"     First feature: {parsed_features[0]}")
            except Exception as e:
                print(f"     Failed to parse features: {e}")
    
    # Check evidences
    print("\n3. Checking evidences file...")
    try:
        evidences = json.load(open("release_evidences.json"))
        print(f"   Evidences loaded: {len(evidences)} items")
        print(f"   Sample evidence keys: {list(evidences.keys())[:5]}")
    except Exception as e:
        print(f"   Failed to load evidences: {e}")
        return
    
    # Test symptom processing
    print("\n4. Testing symptom processing...")
    male_cases = [case for case in cases if case.get('sex') == 'M'][:10]
    
    symptoms_found = []
    for i, case in enumerate(male_cases):
        features = case.get('features', [])
        
        # Convert string to list if needed
        if isinstance(features, str):
            try:
                import ast
                features = ast.literal_eval(features)
            except:
                features = []
        
        if features and len(features) > 0:
            # Get symptom text
            symptom_texts = []
            for feature in features[:3]:  # First 3 features
                if isinstance(feature, str):
                    symptom_name = feature.split("_@_")[0] if "_@_" in feature else feature
                    if symptom_name in evidences:
                        evidence_dict = evidences[symptom_name]
                        symptom_text = evidence_dict.get('question_en', f'[{symptom_name}]')
                        symptom_texts.append(symptom_text)
                    else:
                        symptom_texts.append(f"[{symptom_name}]")
            
            combined_symptoms = ", ".join(symptom_texts) if symptom_texts else "chest pain"
            symptoms_found.append(combined_symptoms)
            
            if i < 5:
                print(f"   Case {i} symptoms: {combined_symptoms[:80]}...")
    
    # Check diversity
    unique_symptoms = len(set(symptoms_found))
    print(f"\n5. Symptom diversity check:")
    print(f"   Total cases processed: {len(symptoms_found)}")
    print(f"   Unique symptom combinations: {unique_symptoms}")
    print(f"   Diversity ratio: {unique_symptoms/len(symptoms_found):.2f}")
    
    if unique_symptoms < len(symptoms_found) * 0.5:
        print("   ⚠️  LOW DIVERSITY DETECTED!")
        print("   Most common symptoms:")
        from collections import Counter
        symptom_counts = Counter(symptoms_found)
        for symptom, count in symptom_counts.most_common(3):
            print(f"     '{symptom[:50]}...' appears {count} times")
    
    # Test actual prompt generation
    print("\n6. Testing actual prompt generation...")
    try:
        from src.advai.data.prompt_builder import PromptBuilder
        
        conditions_mapping = json.load(open("release_conditions.json"))
        
        prompt_builder = PromptBuilder(
            conditions_mapping,
            demographic_concepts=["sex"],
            evidences=evidences,
            concepts_to_test=["sex"],
            full_prompt_template="A {{ sex|clean }} patient has symptoms: {{ symptoms }}.",
            baseline_prompt_template="A patient has symptoms: {{ symptoms }}."
        )
        
        print("   Testing PromptBuilder on 5 male cases:")
        for i, case in enumerate(male_cases[:5]):
            try:
                prompt_with, symptoms = prompt_builder.build_prompts(case, i, ('sex',))
                prompt_without, _ = prompt_builder.build_prompts(case, i, ())
                
                print(f"   Case {i}:")
                print(f"     With sex: {prompt_with[:80]}...")
                print(f"     Without:  {prompt_without[:80]}...")
                print(f"     Symptoms: {symptoms[:60] if symptoms else 'None'}...")
                
            except Exception as e:
                print(f"     Error in case {i}: {e}")
        
    except Exception as e:
        print(f"   Failed to test PromptBuilder: {e}")
    
    print("\n✅ Debug complete!")


if __name__ == "__main__":
    debug_prompt_generation()

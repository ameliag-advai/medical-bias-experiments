import argparse
import os
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.pipeline import run_analysis_pipeline
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Bias Analysis Pipeline")
    parser.add_argument("--output", type=str, help="Path to write output file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma", "gpt2"],
        default="gemma",
        help="Model to use",
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], help="Device to run the model on"
    )
    parser.add_argument("--patient-file", type=str, help="Path to patient data CSV")
    parser.add_argument("--num-cases", type=int, help="Number of cases to process")
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["age", "sex"],
        help="Concepts to analyze",
    )
    args = parser.parse_args()

    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    model, sae = load_model_and_sae(model_scope=args.model, device=args.device)
    conditions_path = "release_conditions.json" # "/mnt/advai_scratch/shared/data/ddxdataset/release_conditions.json"
    patient_path = args.patient_file or "release_test_patients" # "/mnt/advai_scratch/shared/data/ddxdataset/release_test_patients"

    run_analysis_pipeline(
        patient_data_path=patient_path,
        conditions_json_path=conditions_path,
        model=model,
        sae=sae,
        num_cases=args.num_cases,
        concepts_to_test=args.concepts,
        output_name=args.output,
    )


if __name__ == "__main__":
    main()

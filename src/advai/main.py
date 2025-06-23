import argparse
import os
import re
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.pipeline import run_analysis_pipeline
from dotenv import load_dotenv


def validate_device(device_str):
    """Validate device string format."""
    if device_str == "cpu":
        return device_str
    elif device_str == "cuda":
        return device_str
    elif re.match(r'^cuda:\d+$', device_str):
        return device_str
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid device '{device_str}'. Must be 'cpu', 'cuda', or 'cuda:N' where N is the GPU index (e.g., 'cuda:0', 'cuda:1')"
        )


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
        "--device", 
        type=validate_device,  # Use custom validation instead of choices
        default="cuda",
        help="Device to run the model on. Options: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc."
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
    parser.add_argument("--start-case", type=int, default=0, help="Start case index for analysis. Enter an integer value between 0 and 134529.")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Optionally print which device is being used
    print(f"Using device: {args.device}")

    model, sae = load_model_and_sae(model_scope=args.model, device=args.device)
    conditions_path = "release_conditions.json"
    evidences_path = "release_evidences.json"
    patient_path = args.patient_file or "release_test_patients"

    run_analysis_pipeline(
        patient_data_path=patient_path,
        conditions_json_path=conditions_path,
        evidences_json_path=evidences_path,
        model=model,
        sae=sae,
        num_cases=args.num_cases,
        start_case=args.start_case,
        concepts_to_test=args.concepts,
        output_name=args.output,
    )


if __name__ == "__main__":
    main()
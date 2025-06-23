import argparse
import os
import re
from dotenv import load_dotenv

from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.pipeline import run_analysis_pipeline
from src.advai.analysis.clamping_analysis import main as clamping_main


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


def parse_feature_group(group_str):
    """Parse a feature group from the command line."""
    return group_str.lower().strip().split()


def main():
    parser = argparse.ArgumentParser(description="Bias Analysis Pipeline")
    parser.add_argument("--output", type=str, help="Path to write output file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma"],
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
    parser.add_argument('--clamp', action='store_true', help='Enable clamping feature')
    parser.add_argument(
        '--clamp-features',
        type=parse_feature_group,
        nargs='+',
        choices=[
            ['male'],
            ['female'],
            ['old'],
            ['young'],
            ['old', 'male'],
            ['old', 'female'],
            ['young', 'male'],
            ['young', 'female']
        ],
        help=(
            "Groups of features to clamp. Ensure to separate groups containing more than one feature with spaces. \
            For example: --clamp-features 'male old' 'female young' 'old'. The choice ['male'] corresponds to 'male' in \
            the terminal command. The choice ['old', 'male'] corresponds to 'old male' in the terminal command."
        )
    )
    parser.add_argument('--clamp-values', type=int, nargs="+", choices=[0, 5, 10], help='Values to clamp the feature at')
    parser.add_argument('--post-hoc-analysis', action='store_true', help='Run post-hoc clamping analysis on existing results')
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Optionally print which device is being used
    print(f"Using device: {args.device}")

    # If post-hoc analysis requested, invoke clamping_analysis
    if args.post_hoc_analysis:
        if not args.clamp_features or args.clamp_values is None:
            raise ValueError("Clamp feature and value must be specified when post-hoc analysis is enabled.")
        import sys as _sys
        _sys.argv = [_sys.argv[0], '--demographic', args.clamp_features, '--extent', str(args.clamp_values)]
        clamping_main()
        return

    # Validate clamping
    if args.clamp and (not args.clamp_features or not args.clamp_values):
        raise ValueError("When clamping is enabled, both --clamp-features and --clamp-value must be specified.")

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
        clamping=args.clamp,
        clamp_features=args.clamp_features if args.clamp else None,
        clamp_values=args.clamp_values if args.clamp else None,
    )


if __name__ == "__main__":
    main()

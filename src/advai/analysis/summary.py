"""Generate a summary of the analysis results."""


def generate_summary(results):
    """Generate a human-readable summary of the analysis."""
    if len(results) == 0:
        raise ValueError("No results to summarize.")

    total_cases = len(results)
    n_changed = sum(1 for r in results if r["n_active_1"] != r["n_active_2"])

    lines = [
        "=== SUMMARY ===",
        f"Total cases: {total_cases}",
        f"Diagnosis changed (activation count): {n_changed}/{total_cases}",
    ]

    return "\n".join(lines)


def write_output(output_path, case_summaries, summary_text):
    """
    Write the analysis results to an output file.
    Args:
        output_path: Path to write the output file
        case_summaries: List of case summary strings
        summary_text: The summary text to write
    Returns:
        str: The path to the output file
    """
    with open(output_path, "w") as f:
        for idx, summary in enumerate(case_summaries):
            f.write(f"--- CASE {idx+1} ---\n{summary}\n\n")
        f.write("\n" + summary_text + "\n")

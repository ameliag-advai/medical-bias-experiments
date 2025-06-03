def generate_summary(results, case_summaries, activation_diff_by_sex, activation_diff_by_diagnosis):
    """
    Generate a summary of the analysis results.
    Args:
        results: List of analysis results
        case_summaries: List of case summary strings
        activation_diff_by_sex: Dictionary of activation differences by sex
        activation_diff_by_diagnosis: Dictionary of activation differences by diagnosis
    Returns:
        str: The summary text
    """
    total_cases = len(results)
    n_changed = sum(1 for r in results if r.get('n_active_with') != r.get('n_active_without'))
    lines = [
        "=== SUMMARY ===",
        f"Total cases: {total_cases}",
        f"Diagnosis changed (activation count): {n_changed}/{total_cases}",
        "",
        "--- Activation Differences by Sex ---",
    ]
    for sex, diff in activation_diff_by_sex.items():
        lines.append(f"  {sex}: {diff}")
    lines.append("")
    lines.append("--- Activation Differences by Diagnosis ---")
    for dx, diff in activation_diff_by_diagnosis.items():
        lines.append(f"  {dx}: {diff}")
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
    return output_path
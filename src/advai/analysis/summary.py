"""Generate a summary of the analysis results."""


def generate_summary(results, pairs_to_compare) -> str:
    """Generate a human-readable summary of the analysis.
    
    :param results: List of dictionaries containing results for each case.
    :param pairs_to_compare: List of pairs of demographic combinations to compare.
    :return: A string containing the summary of the analysis.
    """
    if len(results) == 0:
        raise ValueError("No results to summarize.")
    
    lines = ["=== SUMMARY ==="]

    for pair in pairs_to_compare:
        results_for_pair = [r[pair] for r in results if r[pair] is not None]

        total_cases = len(results_for_pair)
        n_changed = sum(1 for r in results_for_pair if r["n_active_1"] != r["n_active_2"])

        lines.append(f"\n--- Comparison: {pair} ---")
        lines.append(f"Total cases where demographic exists: {total_cases}")
        lines.append(f"Cases with changed diagnosis (activation count): {n_changed}/{total_cases}\n")

        if "sex" in pair:
            mean_activation_diff = sum(r["activation_difference"] for r in results_for_pair) / total_cases
            lines.append("--- Activation Difference by Sex ---")
            lines.append(f"Mean activation difference: {mean_activation_diff:.4f}")
        
        elif "age" in pair:
            mean_activation_diff = sum(r["activation_difference"] for r in results_for_pair) / total_cases
            lines.append("--- Activation Difference by Age ---")
            lines.append(f"Mean activation difference: {mean_activation_diff:.4f}")

        elif "age_sex" in pair or "sex_age" in pair:
            mean_activation_diff = sum(r["activation_difference"] for r in results_for_pair) / total_cases
            lines.append("--- Activation Difference by Age and Sex ---")
            lines.append(f"Mean activation difference: {mean_activation_diff:.4f}")
        
    return "\n".join(lines)


def write_output(output_path, case_summaries, summary_text):
    """Write the full analysis results to disk.
    
    :param output_path: Path to save the output file.
    :param case_summaries: List of summaries for each case.
    :param summary_text: Summary text generated from the analysis.
    """
    with open(output_path, "w") as f:
        for idx, summary in enumerate(case_summaries):
            f.write(f"--- CASE {idx+1} ---\n{summary}\n\n")
        f.write("\n" + summary_text + "\n")

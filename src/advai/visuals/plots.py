import plotly.graph_objects as go


# def visualize_feature_overlaps(results, pairs_to_compare, save_path="feature_overlap.html"):
#     """Generate a visualization of feature overlap and activation differences."""
#     diffs = [r['activation_difference'] for r in results if r.get('activation_difference') is not None]

#     fig = go.Figure(
#         data=[go.Bar(y=diffs, x=[f"Case {i+1}" for i in range(len(diffs))], name="Activation Δ (L2)")]
#     )
#     fig.update_layout(
#         title="Activation Differences Across Cases",
#         xaxis_title="Case",
#         yaxis_title="L2 Distance"
#     )
#     fig.write_html(save_path)


def visualize_feature_overlaps(results, pairs_to_compare, save_path="feature_overlap.html"):
    """Generate a visualization of feature overlap and activation differences."""

    for pair_to_compare in pairs_to_compare:
        clean_results_for_this_pair = []
        clean_x_for_this_pair = []
        for i, r in enumerate(results):
            if r[pair_to_compare] is not None:
                clean_results_for_this_pair.append(r[pair_to_compare])
                clean_x_for_this_pair.append(f"Case {r[pair_to_compare]['case_id']}")

        if len(clean_results_for_this_pair) == 0:
            raise ValueError("No results to summarize.")

        clean_diffs = [r["activation_difference"] for r in clean_results_for_this_pair if r["activation_difference"] is not None]

        fig = go.Figure(
            data=[go.Bar(y=clean_diffs, x=clean_x_for_this_pair, name="Activation Δ (L2)")]
        )

        pair_title = pair_to_compare[0].split("_")
        comparison = "Comparing prompts with Demographics (" + ", ".join(pair_title) + ") vs. No Demographics"

        fig.update_layout(
            title="Activation Differences Across Cases. " + comparison,
            xaxis_title="Case",
            yaxis_title="L2 Distance"
        )

        comp1 = pair_to_compare[0]
        comp2 = pair_to_compare[1] if len(pair_to_compare) > 1 else "no_demo"
        pair_save_path = save_path + "_comparing_" + comp1 + "_vs_" + comp2 + ".html"
        fig.write_html(pair_save_path)

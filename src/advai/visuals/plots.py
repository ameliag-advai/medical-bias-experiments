import plotly.graph_objects as go

def visualize_feature_overlaps(results, save_path="feature_overlap.html"):
    """Generate a visualization of feature overlap and activation differences."""
    diffs = [r['activation_difference'] for r in results if r.get('activation_difference') is not None]

    fig = go.Figure(
        data=[go.Bar(y=diffs, x=[f"Case {i+1}" for i in range(len(diffs))], name="Activation Δ (L2)")]
    )
    fig.update_layout(
        title="Activation Differences Across Cases",
        xaxis_title="Case",
        yaxis_title="L2 Distance"
    )
    fig.write_html(save_path)
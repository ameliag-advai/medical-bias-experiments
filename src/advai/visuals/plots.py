import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
from datetime import datetime


def _get_timestamped_filename(base_name, ext=".html"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name.endswith(ext):
        base_name = base_name[: -len(ext)]
    return f"{base_name}_{timestamp}{ext}"


def visualize_feature_overlaps(results, pairs_to_compare, save_path="feature_overlap.html"):
    """Generate a visualization of feature overlap and activation differences.
    
    :param results: List of dictionaries containing results for each case.
    :param pairs_to_compare: List of tuples, each containing two group names to compare.
    :param save_path: Path to save the generated HTML file. If None, a timestamped filename is created.
    """
    if save_path is None:
        save_path = _get_timestamped_filename("feature_overlap")

    for pair_to_compare in pairs_to_compare:
        group1_name = pair_to_compare[0]
        group2_name = pair_to_compare[1] if len(pair_to_compare[1]) > 0 else "no_demo"
        clean_results_for_this_pair = []
        clean_x_for_this_pair = []
        for i, r in enumerate(results):
            if r[pair_to_compare] is not None:
                clean_results_for_this_pair.append(r[pair_to_compare])
                clean_x_for_this_pair.append(f"Case {r[pair_to_compare]['case_id']}")

        if len(clean_results_for_this_pair) == 0:
            raise ValueError("No results to summarize.")

        # Insert table making here.
        table_rows = []
        feature_diff_rows = []
        for r in clean_results_for_this_pair:
            top_1 = r.get("top_dxs_" + group1_name, [])
            top_2 = r.get("top_dxs_" + group2_name, [])
            print(f"Top 1: {top_1}, Top 2: {top_2}")
            if top_1 and top_2 and (top_1[0] != top_2[0]):
                ci = r.get("case_info", {})
                table_rows.append([
                    ci.get("age", ""),
                    ci.get("sex", ""),
                    top_1[0],
                    top_2[0]
                ])
                features_1 = r.get("features_" + group1_name, [])
                features_2 = r.get("features_" + group2_name, [])
                diff_1 = [i for i, x in enumerate(features_1) if x and not features_2[i]]
                diff_2 = [i for i, x in enumerate(features_2) if x and not features_1[i]]
                feature_diff_rows.append([
                    ci.get("age", ""),
                    ci.get("sex", ""),
                    top_1[0],
                    top_2[0],
                    ', '.join(map(str, diff_1)),
                    ', '.join(map(str, diff_2))
                ])

        # Create the figure with subplots
        pair_title = pair_to_compare[0].split("_")
        comparison = "Comparing prompts with Demographics (" + ", ".join(pair_title) + ") vs. No Demographics"
        fig = sp.make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                "Average Change in Feature Activation (L2). " + comparison, 
                f"Cases Where Top-1 Diagnosis Changed (Group 1: {group1_name} vs. Group 2: {group2_name})", 
                "Summary of Feature Differences"],
            row_heights=[0.3, 0.3, 0.4],
            specs=[[{"type": "bar"}], [{"type": "table"}], [{"type": "table"}]]
        )

        # Plot the average change in feature activation
        clean_diffs = [r["activation_difference"] for r in clean_results_for_this_pair if r["activation_difference"] is not None]
        if clean_diffs:
            fig.add_trace(
                go.Bar(y=clean_diffs, x=clean_x_for_this_pair, name="Activation Δ (L2)"),
                row=1, col=1
            )
        else:
            fig.add_trace(go.Bar(y=[], x=[]), row=1, col=1)

        # Create a table of cases where the top-1 diagnosis changed
        if table_rows:
            fig.add_trace(
                go.Table(
                    header=dict(values=["Age", "Sex", "Dx (" + group1_name + ")", "Dx (" + group2_name + ")"]),
                    cells=dict(values=list(zip(*table_rows)))
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(
                go.Table(
                    header=dict(values=["Age", "Sex", "Dx (" + group1_name + ")", "Dx (" + group2_name + ")"]),
                    cells=dict(values=[[], [], [], []])
                ),
                row=2, col=1
            )

        # Create a table of feature differences
        if feature_diff_rows:
            fig.add_trace(
                go.Table(
                    header=dict(values=["Age", "Sex", "Dx (" + group1_name + ")", "Dx (" + group2_name + ")", 
                                    "Features Active with " + group1_name + " but not " + group2_name,
                                    "Features Active with " + group2_name + " but not " + group1_name]),
                    cells=dict(values=list(zip(*feature_diff_rows)))
                ),
                row=3, col=1
            )
        else:
            fig.add_trace(
                go.Table(
                    header=dict(values=["Age", "Sex", "Dx (" + group1_name + ")", "Dx (" + group2_name + ")",
                                    "Features Active with " + group1_name + " but not " + group2_name,
                                    "Features Active with " + group2_name + " but not " + group1_name]),
                    cells=dict(values=[[], [], [], [], [], []])
                ),
                row=3, col=1
            )        

        # Save the figure to an HTML file
        comp1 = pair_to_compare[0]
        comp2 = pair_to_compare[1] if len(pair_to_compare) > 1 else "no_demo"
        pair_save_path = save_path + "_comparing_" + comp1 + "_vs_" + comp2 + ".html"
        fig.update_layout(height=1000, showlegend=False)
        fig.write_html(pair_save_path)


def visualize_clamping_analysis(csv_path, save_path=None):
    """Visualize the output of clampinganalysis.csv, showing for each case if features or diagnoses changed between groups.
    
    Generates an interactive HTML table (plotly) highlighting changes. If save_path is None, a timestamped file is created.

    :param csv_path: Path to the CSV file containing clamping analysis results.
    :param save_path: Path to save the generated HTML file. If None, a timestamped filename is created.
    """
    if save_path is None:
        save_path = _get_timestamped_filename("clamping_feature_diag_changes")
    df = pd.read_csv(csv_path)
    groups = ['with_demo', 'no_demo', 'clamped']
    cases = sorted(df['case_id'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    table_rows = []
    for case_id in cases:
        row = [case_id]
        for group in groups:
            sub = df[(df['case_id']==case_id)&(df['group']==group)]
            if not sub.empty:
                feats = sub.iloc[0]['top_5_features']
                diags = sub.iloc[0]['top_5_diagnoses']
                feats_changed = sub.iloc[0]['features_changed']
                diags_changed = sub.iloc[0]['diagnoses_changed']
                cell = f"Features: {feats}<br>Diagnoses: {diags}"
                if feats_changed or diags_changed:
                    cell += f"<br><b>CHANGED</b>"
            else:
                cell = ""
            row.append(cell)
        table_rows.append(row)
    headers = ["Case ID"] + [f"{g}" for g in groups]
    fig = go.Figure(
        data=[go.Table(
            header=dict(values=headers),
            cells=dict(values=list(zip(*table_rows)))
        )]
    )
    fig.update_layout(height=400 + 30*len(cases), showlegend=False)
    fig.write_html(save_path)
    print(f"[INFO] Clamping analysis visualization saved to {save_path}")
    print(f'[DEBUG] feature_overlap.html written to {save_path}')

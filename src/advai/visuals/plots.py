import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
from datetime import datetime


def _get_timestamped_filename(base_name, ext=".html"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name.endswith(ext):
        base_name = base_name[: -len(ext)]
    return f"{base_name}_{timestamp}{ext}"


def visualize_feature_overlaps(results, save_path=None):
    """
    Visualize average change in feature activation and show a table of cases where the top-1 dx changed.
    Args:
        results: List of dicts from analyze_case_for_bias, one per case.
        save_path: Where to save the HTML file. If None, a timestamped file is created.
    """
    if save_path is None:
        save_path = _get_timestamped_filename("feature_overlap")
    diffs = [r.get('activation_difference', None) for r in results]
    diffs = [d for d in diffs if d is not None]
    table_rows = []
    feature_diff_rows = []
    for r in results:
        top_with = r.get('top_dxs_with_demo', [])
        top_without = r.get('top_dxs_without_demo', [])
        if top_with and top_without and top_with[0] != top_without[0]:
            ci = r.get('case_info', {})
            table_rows.append([
                ci.get('age', ''),
                ci.get('sex', ''),
                top_with[0],
                top_without[0]
            ])
            with_demo = r.get('features_with_demo', [])
            without_demo = r.get('features_without_demo', [])
            diff_with = [i for i, x in enumerate(with_demo) if x and not without_demo[i]]
            diff_without = [i for i, x in enumerate(without_demo) if x and not with_demo[i]]
            feature_diff_rows.append([
                ci.get('age', ''),
                ci.get('sex', ''),
                top_with[0],
                top_without[0],
                ', '.join(map(str, diff_with)),
                ', '.join(map(str, diff_without))
            ])

    fig = sp.make_subplots(
        rows=3, cols=1,
        subplot_titles=["Average Change in Feature Activation (L2)", "Cases Where Top-1 Diagnosis Changed (Demo vs No Demo)", "Summary of Feature Differences"],
        row_heights=[0.3, 0.3, 0.4],
        specs=[[{"type": "bar"}], [{"type": "table"}], [{"type": "table"}]]
    )

    if diffs:
        fig.add_trace(
            go.Bar(y=diffs, x=[f"Case {i+1}" for i in range(len(diffs))], name="Activation Δ (L2)"),
            row=1, col=1
        )
    else:
        fig.add_trace(go.Bar(y=[]), row=1, col=1)
    if table_rows:
        fig.add_trace(
            go.Table(
                header=dict(values=["Age", "Sex", "Dx (Demo)", "Dx (No Demo)"]),
                cells=dict(values=list(zip(*table_rows)))
            ),
            row=2, col=1
        )
    else:
        fig.add_trace(
            go.Table(
                header=dict(values=["Age", "Sex", "Dx (Demo)", "Dx (No Demo)"]),
                cells=dict(values=[[], [], [], []])
            ),
            row=2, col=1
        )
    if feature_diff_rows:
        fig.add_trace(
            go.Table(
                header=dict(values=["Age", "Sex", "Dx (Demo)", "Dx (No Demo)", "Features Active with Demo but not Without", "Features Active without Demo but not With"]),
                cells=dict(values=list(zip(*feature_diff_rows)))
            ),
            row=3, col=1
        )
    else:
        fig.add_trace(
            go.Table(
                header=dict(values=["Age", "Sex", "Dx (Demo)", "Dx (No Demo)", "Features Active with Demo but not Without", "Features Active without Demo but not With"]),
                cells=dict(values=[[], [], [], [], [], []])
            ),
            row=3, col=1
        )
    fig.update_layout(height=1000, showlegend=False)
    fig.write_html(save_path)


def visualize_feature_overlaps(results, pairs_to_compare, save_path="feature_overlap.html"):
    """Generate a visualization of feature overlap and activation differences."""
    if save_path is None:
        save_path = _get_timestamped_filename("feature_overlap")

    for pair_to_compare in pairs_to_compare:
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

        # Create the figure with subplots
        pair_title = pair_to_compare[0].split("_")
        comparison = "Comparing prompts with Demographics (" + ", ".join(pair_title) + ") vs. No Demographics"
        fig = sp.make_subplots(
            rows=3, cols=1,
            subplot_titles=["Average Change in Feature Activation (L2). " + comparison, "Cases Where Top-1 Diagnosis Changed (Demo vs No Demo)", "Summary of Feature Differences"],
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


        # Create a table of feature differences
        

        # Save the figure to an HTML file
        comp1 = pair_to_compare[0]
        comp2 = pair_to_compare[1] if len(pair_to_compare) > 1 else "no_demo"
        pair_save_path = save_path + "_comparing_" + comp1 + "_vs_" + comp2 + ".html"
        fig.update_layout(height=1000, showlegend=False)
        fig.write_html(pair_save_path)


def visualize_clamping_analysis(csv_path, save_path=None):
    """
    Visualize the output of clampinganalysis.csv, showing for each case if features or diagnoses changed between groups.
    Generates an interactive HTML table (plotly) highlighting changes.
    If save_path is None, a timestamped file is created.
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

import plotly.graph_objects as go
import plotly.subplots as sp

def visualize_feature_overlaps(results, save_path="feature_overlap.html"):
    """
    Visualize average change in feature activation and show a table of cases where the top-1 dx changed.
    Args:
        results: List of dicts from analyze_case_for_bias, one per case.
        save_path: Where to save the HTML file.
    """
    # Gather per-case activation difference
    diffs = [r.get('activation_difference', None) for r in results]
    diffs = [d for d in diffs if d is not None]
    # Gather table info for dx-changed cases
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
    # Create subplots: bar for avg activation diff, table for dx-changed cases, summary table
    fig = sp.make_subplots(
        rows=3, cols=1,
        subplot_titles=["Average Change in Feature Activation (L2)", "Cases Where Top-1 Diagnosis Changed (Demo vs No Demo)", "Summary of Feature Differences"],
        row_heights=[0.3, 0.3, 0.4],
        specs=[[{"type": "bar"}], [{"type": "table"}], [{"type": "table"}]]
    )
    # Bar plot
    if diffs:
        fig.add_trace(
            go.Bar(y=diffs, x=[f"Case {i+1}" for i in range(len(diffs))], name="Activation Δ (L2)"),
            row=1, col=1
        )
    else:
        fig.add_trace(go.Bar(y=[]), row=1, col=1)
    # Table
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
    # Summary table
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


def visualize_clamping_analysis(csv_path, save_path="clamping_feature_diag_changes.html"):
    """
    Visualize the output of clampinganalysis.csv, showing for each case if features or diagnoses changed between groups.
    Generates an interactive HTML table (plotly) highlighting changes.
    """
    import pandas as pd
    import plotly.graph_objects as go
    df = pd.read_csv(csv_path)
    # Pivot so each row is a case, columns for each group
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

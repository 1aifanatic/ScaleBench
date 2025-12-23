"""ScaleDown Compression Leaderboard - Gradio App.

Displays benchmark results comparing baseline vs ScaleDown context compression.
Supports dynamic metrics discovered from uploaded results.
"""
import gradio as gr
import pandas as pd

from src.about import (
    TITLE,
    INTRODUCTION_TEXT,
    ABOUT_TEXT,
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
)
from src.display.css import custom_css
from src.display.columns import (
    build_column_config,
    get_column_names,
    get_default_columns,
    get_never_hidden_columns,
    get_column_types,
)
from src.display.formatting import styled_error
from src.data_loader import get_leaderboard_df, get_comparison_df
from src.envs import API, RESULTS_DATASET, TOKEN


def load_data():
    """Load data and build column configuration."""
    df, metric_cols = get_leaderboard_df()
    columns = build_column_config(metric_cols)
    return df, columns, metric_cols


def init_leaderboard(df: pd.DataFrame, columns: list):
    """Initialize the main leaderboard component.
    
    Uses gr.Dataframe since gradio_leaderboard may not be available.
    """
    if df is None or df.empty:
        return gr.Dataframe(
            value=pd.DataFrame({"Status": ["No results found. Upload benchmark results to see the leaderboard."]}),
            interactive=False,
        )
    
    # Select columns that exist in the dataframe
    col_names = get_column_names(columns)
    available_cols = [c for c in col_names if c in df.columns]
    
    display_df = df[available_cols].copy() if available_cols else df
    
    return gr.Dataframe(
        value=display_df,
        datatype=get_column_types([c for c in columns if c.name in available_cols]),
        interactive=False,
        wrap=True,
    )


def init_comparison_table(df: pd.DataFrame):
    """Initialize baseline vs scaledown comparison table."""
    comparison_df = get_comparison_df(df)
    
    if comparison_df.empty:
        return gr.Dataframe(
            value=pd.DataFrame({"Status": ["No paired baseline/scaledown results found for comparison."]}),
            interactive=False,
        )
    
    # Select key comparison columns
    display_cols = ["model", "dataset"]
    
    # Add metric columns (baseline, scaledown, delta)
    for col in comparison_df.columns:
        if col.endswith("_baseline") or col.endswith("_scaledown") or col.endswith("_delta"):
            if "_mean_" in col or col.endswith("_mean_baseline") or col.endswith("_mean_scaledown") or col.endswith("_mean_delta"):
                display_cols.append(col)
    
    # Add cost savings
    if "cost_savings_pct" in comparison_df.columns:
        display_cols.append("cost_savings_pct")
    
    available_cols = [c for c in display_cols if c in comparison_df.columns]
    display_df = comparison_df[available_cols] if available_cols else comparison_df
    
    return gr.Dataframe(
        value=display_df,
        interactive=False,
        wrap=True,
    )


def filter_dataframe(df: pd.DataFrame, model: str, dataset: str, mode: str) -> pd.DataFrame:
    """Filter dataframe based on selections."""
    filtered = df.copy()
    
    if model and model != "All":
        filtered = filtered[filtered["model"] == model]
    if dataset and dataset != "All":
        filtered = filtered[filtered["dataset"] == dataset]
    if mode and mode != "All":
        filtered = filtered[filtered["mode"] == mode]
    
    return filtered


def create_demo():
    """Create the Gradio demo app."""
    # Load data
    df, columns, metric_cols = load_data()
    
    # Get unique values for filters
    models = ["All"] + sorted(df["model"].unique().tolist()) if not df.empty and "model" in df.columns else ["All"]
    datasets = ["All"] + sorted(df["dataset"].unique().tolist()) if not df.empty and "dataset" in df.columns else ["All"]
    modes = ["All", "baseline", "scaledown"]
    
    with gr.Blocks(css=custom_css) as demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")
        
        with gr.Tabs(elem_classes="tab-buttons") as tabs:
            # Tab 1: All Results
            with gr.TabItem("All Results", elem_id="all-results-tab", id=0):
                with gr.Row():
                    model_filter = gr.Dropdown(
                        choices=models,
                        value="All",
                        label="Model",
                        interactive=True,
                    )
                    dataset_filter = gr.Dropdown(
                        choices=datasets,
                        value="All",
                        label="Dataset",
                        interactive=True,
                    )
                    mode_filter = gr.Dropdown(
                        choices=modes,
                        value="All",
                        label="Mode",
                        interactive=True,
                    )
                
                leaderboard = init_leaderboard(df, columns)
                
                # Filter functionality
                def update_leaderboard(model, dataset, mode):
                    filtered = filter_dataframe(df, model, dataset, mode)
                    col_names = get_column_names(columns)
                    available_cols = [c for c in col_names if c in filtered.columns]
                    return filtered[available_cols] if available_cols else filtered
                
                model_filter.change(
                    fn=update_leaderboard,
                    inputs=[model_filter, dataset_filter, mode_filter],
                    outputs=leaderboard,
                )
                dataset_filter.change(
                    fn=update_leaderboard,
                    inputs=[model_filter, dataset_filter, mode_filter],
                    outputs=leaderboard,
                )
                mode_filter.change(
                    fn=update_leaderboard,
                    inputs=[model_filter, dataset_filter, mode_filter],
                    outputs=leaderboard,
                )
            
            # Tab 2: Baseline vs ScaleDown Comparison
            with gr.TabItem("Baseline vs ScaleDown", elem_id="comparison-tab", id=1):
                gr.Markdown("""
                ### Side-by-side comparison of baseline and ScaleDown results
                
                Shows paired experiments with the same model and dataset, comparing quality metrics and cost savings.
                """)
                comparison_table = init_comparison_table(df)
            
            # Tab 3: About
            with gr.TabItem("About", elem_id="about-tab", id=2):
                gr.Markdown(ABOUT_TEXT, elem_classes="markdown-text")
                
                with gr.Accordion("Citation", open=False):
                    gr.Textbox(
                        value=CITATION_BUTTON_TEXT,
                        label=CITATION_BUTTON_LABEL,
                        lines=6,
                        elem_id="citation-button",
                    )
    
    return demo


# Create and launch the demo
demo = create_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )

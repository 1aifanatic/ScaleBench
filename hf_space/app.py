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


def load_data(force_refresh: bool = False):
    """Load data and build column configuration.
    
    Args:
        force_refresh: If True, bypass cache and fetch fresh data from HF.
    """
    df, metric_cols = get_leaderboard_df(force_download=force_refresh)
    columns = build_column_config(metric_cols)
    return df, columns, metric_cols


def refresh_data():
    """Refresh data from HuggingFace dataset.
    
    Forces re-download bypassing cache.
    
    Returns:
        Tuple of (main_leaderboard, comparison_table, status_message)
    """
    try:
        df, columns, _ = load_data(force_refresh=True)
        leaderboard = init_leaderboard(df, columns)
        comparison = init_comparison_table(df)
        return leaderboard, comparison, "✓ Data refreshed successfully"
    except Exception as e:
        error_msg = f"Error refreshing data: {str(e)}"
        return None, None, error_msg


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
    
    # Select key columns for display
    # Start with model and dataset
    display_cols = []
    for col in comparison_df.columns:
        if col in ["Model", "Dataset"]:
            display_cols.append(col)
    
    # Add metric columns (baseline, scaledown, delta)
    for col in comparison_df.columns:
        if "(Baseline)" in col or "(ScaleDown)" in col or "(Δ)" in col or "Cost Savings" in col:
            display_cols.append(col)
    
    available_cols = [c for c in display_cols if c in comparison_df.columns]
    display_df = comparison_df[available_cols] if available_cols else comparison_df
    
    return gr.Dataframe(
        value=display_df,
        interactive=False,
        wrap=True,
    )


def filter_dataframe(df: pd.DataFrame, model: str, dataset: str, mode: str) -> pd.DataFrame:
    """Filter dataframe based on selections.
    
    Args:
        df: DataFrame with display column names.
        model: Model filter value.
        dataset: Dataset filter value.
        mode: Mode filter value.
        
    Returns:
        Filtered DataFrame.
    """
    filtered = df.copy()
    
    # Use display column names
    if model and model != "All" and "Model" in filtered.columns:
        filtered = filtered[filtered["Model"] == model]
    if dataset and dataset != "All" and "Dataset" in filtered.columns:
        filtered = filtered[filtered["Dataset"] == dataset]
    if mode and mode != "All" and "Mode" in filtered.columns:
        filtered = filtered[filtered["Mode"] == mode]
    
    return filtered


def create_demo():
    """Create the Gradio demo app."""
    # Load data
    df, columns, metric_cols = load_data()
    
    # Get unique values for filters (using display column names)
    models = ["All"] + sorted(df["Model"].unique().tolist()) if not df.empty and "Model" in df.columns else ["All"]
    datasets = ["All"] + sorted(df["Dataset"].unique().tolist()) if not df.empty and "Dataset" in df.columns else ["All"]
    modes = ["All", "baseline", "scaledown"]
    
    with gr.Blocks(css=custom_css) as demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")
        
        # Refresh button at the top
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Data", variant="primary", size="sm")
            refresh_status = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                container=False,
                visible=True,
            )
        
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
                
                **How to read the delta columns:**
                - **Delta** = ScaleDown value - Baseline value
                - **Positive delta**: ScaleDown performs better (e.g., +0.05 for F1 means 5% improvement)
                - **Negative delta**: ScaleDown performs worse (e.g., -0.02 for F1 means 2% degradation)
                - **Cost Savings %**: Percentage reduction in cost when using ScaleDown
                
                **Expected results:**
                - Token savings: 80-90% reduction in input tokens
                - Quality metrics: Should be similar (±5-10% delta is acceptable)
                - Cost savings: Significant reduction proportional to token savings
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
        
        # Wire up refresh button
        refresh_btn.click(
            fn=refresh_data,
            inputs=None,
            outputs=[leaderboard, comparison_table, refresh_status],
        )
    
    return demo


# Create and launch the demo
demo = create_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )

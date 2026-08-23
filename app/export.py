"""app/export.py — CSV and figure export helpers."""

from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from validation import MAX_DOWNLOAD_ROWS


def get_fitted_parameters_df(model_type: str, state: dict) -> pd.DataFrame:
    """Generate a formatted DataFrame containing fitted parameters and units."""
    records = []
    if model_type == "Binary Crowding Model":
        if state.get("fitted_eps") is not None:
            records.append({"Parameter": "eps (Free Energy Soft Interaction)",
                            "Value": state["fitted_eps"], "Unit": "kJ/mol"})
        if state.get("fitted_epsTS") is not None:
            records.append({"Parameter": "epsTS (Entropic Soft Interaction)",
                            "Value": state["fitted_epsTS"], "Unit": "kJ/mol"})
    else:
        if state.get("fitted_eps2") is not None:
            records.append({"Parameter": "eps2 (Cosolute 2 Free Energy Soft Interaction)",
                            "Value": state["fitted_eps2"], "Unit": "kJ/mol"})
        if state.get("fitted_eps3") is not None:
            records.append({"Parameter": "eps3 (Cosolute 3 Free Energy Soft Interaction)",
                            "Value": state["fitted_eps3"], "Unit": "kJ/mol"})
        if state.get("fitted_eps23") is not None:
            records.append({"Parameter": "eps23 (Synergy Free Energy Soft Interaction)",
                            "Value": state["fitted_eps23"], "Unit": "kJ/mol"})
        if state.get("fitted_epsTS2") is not None:
            records.append({"Parameter": "epsTS2 (Cosolute 2 Entropic Soft Interaction)",
                            "Value": state["fitted_epsTS2"], "Unit": "kJ/mol"})
        if state.get("fitted_epsTS3") is not None:
            records.append({"Parameter": "epsTS3 (Cosolute 3 Entropic Soft Interaction)",
                            "Value": state["fitted_epsTS3"], "Unit": "kJ/mol"})
        if state.get("fitted_epsTS23") is not None:
            records.append({"Parameter": "epsTS23 (Synergy Entropic Soft Interaction)",
                            "Value": state["fitted_epsTS23"], "Unit": "kJ/mol"})
    return pd.DataFrame(records)


def get_fitted_parameters_csv(model_type: str, state: dict) -> str:
    """Return CSV string for fitted parameters."""
    df = get_fitted_parameters_df(model_type, state)
    return "" if df.empty else df.to_csv(index=False)


def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    """Convert a Matplotlib figure to raw bytes (PNG, SVG, or PDF)."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def get_model_results_csv(solved_model) -> str:
    """Convert simulated results DataFrame to CSV string."""
    if hasattr(solved_model, "results") and isinstance(solved_model.results, pd.DataFrame):
        df = solved_model.results
        if len(df) > MAX_DOWNLOAD_ROWS:
            st.error(f"Cannot generate CSV: The results contain {len(df):,} rows, exceeding the safe limit of {MAX_DOWNLOAD_ROWS:,} rows. Decrease the grid resolution to download full results.")
            return ""
        return df.to_csv(index=False)
    return ""


def plotly_fig_to_bytes(fig, fmt: str = "png", scale: int = 2) -> bytes:
    """Convert a Plotly figure to raw bytes (PNG or SVG).

    SVG does not require kaleido.
    PNG requires kaleido; falls back to SVG bytes if kaleido is unavailable.
    """
    import plotly.io as pio
    if fmt == "svg":
        return pio.to_image(fig, format="svg")
    try:
        return pio.to_image(fig, format="png", scale=scale)
    except Exception:
        return pio.to_image(fig, format="svg")

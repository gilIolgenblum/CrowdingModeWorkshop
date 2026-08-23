"""
app/styles.py
=============
CSS injection and UI component helpers for the FH Crowding app.
Palette:
  Cream yellow  #FFF9D2
  Warm orange   #FFEBCC
  Light blue    #BFDDF0
  Medium blue   #8CC0EB
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Palette constants (also used by plotting helpers)
# ---------------------------------------------------------------------------
PALETTE_CREAM   = "#FFF9D2"
PALETTE_WARM    = "#FFEBCC"
PALETTE_LBLUE   = "#BFDDF0"
PALETTE_MBLUE   = "#8CC0EB"
PALETTE_DARK    = "#2C3E50"
PALETTE_MID     = "#5D7A8A"

# Plotly trace colours (model curves + contributions)
PLOT_TOTAL_COLOR = "#2C3E50"
PLOT_NU_COLOR    = "#8CC0EB"
PLOT_CHI_COLOR   = "#E07B5A"
PLOT_EPS_COLOR   = "#6BAA75"
PLOT_EXP_COLOR   = "#C0392B"


def inject_css() -> None:
    """Inject global CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        /* ── Page background ── */
        .stApp {{
            background-color: #f7f9fb;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {PALETTE_CREAM};
            border-right: 1px solid #dde3e8;
        }}
        [data-testid="stSidebar"] .stSubheader {{
            color: {PALETTE_DARK};
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
        }}

        /* ── Section headers ── */
        .fh-section-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1rem;
            background: linear-gradient(90deg, {PALETTE_LBLUE} 0%, {PALETTE_WARM} 100%);
            border-radius: 6px;
            margin-bottom: 1rem;
            border-left: 4px solid {PALETTE_MBLUE};
        }}
        .fh-section-header h3 {{
            margin: 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: {PALETTE_DARK};
        }}

        /* ── Info / workflow card ── */
        .fh-card {{
            background: white;
            border: 1px solid #dde3e8;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid {PALETTE_MBLUE};
        }}
        .fh-card-title {{
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: {PALETTE_MID};
            margin-bottom: 0.3rem;
        }}
        .fh-card-value {{
            font-size: 1.4rem;
            font-weight: 600;
            color: {PALETTE_DARK};
        }}
        .fh-card-desc {{
            font-size: 0.78rem;
            color: #7f8c8d;
            margin-top: 0.2rem;
        }}

        /* ── Mode A / B workflow cards ── */
        .fh-mode-card {{
            background: {PALETTE_WARM};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.5rem;
            border: 1px solid #e8d5b5;
        }}
        .fh-mode-card.mode-b {{
            background: {PALETTE_LBLUE};
            border-color: #b0cfe0;
        }}
        .fh-mode-title {{
            font-weight: 600;
            font-size: 0.92rem;
            color: {PALETTE_DARK};
        }}
        .fh-mode-steps {{
            font-size: 0.8rem;
            color: #5a6a72;
            margin-top: 0.3rem;
            line-height: 1.6;
        }}

        /* ── Divider ── */
        hr.fh-divider {{
            border: none;
            border-top: 1px solid #dde3e8;
            margin: 1.2rem 0;
        }}

        /* ── Download buttons — subtle ── */
        .stDownloadButton > button {{
            background: white;
            border: 1px solid {PALETTE_MBLUE};
            color: {PALETTE_DARK};
            font-size: 0.82rem;
            font-weight: 500;
        }}
        .stDownloadButton > button:hover {{
            background: {PALETTE_LBLUE};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, icon: str = "") -> None:
    """Render a styled section header bar."""
    icon_html = f"{icon}&nbsp;" if icon else ""
    st.markdown(
        f'<div class="fh-section-header"><h3>{icon_html}{title}</h3></div>',
        unsafe_allow_html=True,
    )


def param_card(label: str, value: str, description: str = "") -> str:
    """Return HTML for a fitted-parameter card."""
    desc_html = f'<div class="fh-card-desc">{description}</div>' if description else ""
    return (
        f'<div class="fh-card">'
        f'<div class="fh-card-title">{label}</div>'
        f'<div class="fh-card-value">{value}</div>'
        f'{desc_html}'
        f'</div>'
    )


def workflow_banner() -> None:
    """Render the two-mode workflow explanation banner."""
    st.markdown(
        """
        <div style="display:flex; gap:1rem; margin-bottom:1rem;">
          <div class="fh-mode-card" style="flex:1;">
            <div class="fh-mode-title">⚡ Mode A &mdash; Plug-and-Play Simulation</div>
            <div class="fh-mode-steps">
              1. Choose model (sidebar) &nbsp;→&nbsp;
              2. Set parameters &nbsp;→&nbsp;
              3. Run simulation &nbsp;→&nbsp;
              4. Plot &amp; explore &nbsp;→&nbsp;
              5. Download results
            </div>
          </div>
          <div class="fh-mode-card mode-b" style="flex:1;">
            <div class="fh-mode-title">🔬 Mode B &mdash; Fit to Experimental Data</div>
            <div class="fh-mode-steps">
              1. Choose model &nbsp;→&nbsp;
              2. Upload data (below) &nbsp;→&nbsp;
              3. Set initial parameters &nbsp;→&nbsp;
              4. Run fit &nbsp;→&nbsp;
              5. Plot &amp; review diagnostics &nbsp;→&nbsp;
              6. Download
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

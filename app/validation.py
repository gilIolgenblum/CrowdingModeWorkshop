"""app/validation.py — Helpers for validating simulation grid sizes and inputs."""

import streamlit as st

# Thresholds for grid validation
MAX_SAFE_POINTS = 500_000
MAX_ADVANCED_POINTS = 2_500_000

MAX_BINARY_GRID_POINTS = MAX_ADVANCED_POINTS
MAX_TERNARY_GRID_POINTS = 50_000
MAX_SAFE_TERNARY_POINTS = 10_000
MAX_PLOT_POINTS = 100_000
MAX_DOWNLOAD_ROWS = 250_000
MAX_UPLOAD_ROWS_BINARY = 5_000
MAX_UPLOAD_ROWS_TERNARY = 20_000

def estimate_binary_grid_points(phiC_min: float, phiC_max: float, dphiC: float) -> int:
    """Estimate the number of grid points for a 1D binary simulation."""
    if dphiC <= 0:
        return 0
    return int((phiC_max - phiC_min) / dphiC) + 1

def estimate_ternary_grid_points(phi2_min: float, phi2_max: float, dphi2: float, 
                                 phi3_min: float, phi3_max: float, dphi3: float) -> int:
    """Estimate the number of grid points for a 2D ternary simulation."""
    if dphi2 <= 0 or dphi3 <= 0:
        return 0
    n2 = int((phi2_max - phi2_min) / dphi2) + 1
    n3 = int((phi3_max - phi3_min) / dphi3) + 1
    return n2 * n3

def validate_grid_size(n_points: int, is_advanced: bool, num_arrays: int = 10, model_type: str = "binary"):
    """
    Validate that the grid size is within safe memory limits.
    Stops Streamlit execution if limits are exceeded.
    """
    # Rough memory estimate: 8 bytes per float64 * number of arrays * grid size
    mem_bytes = n_points * 8 * num_arrays
    mem_mb = mem_bytes / (1024 * 1024)
    
    if model_type == "binary":
        threshold = MAX_BINARY_GRID_POINTS if is_advanced else MAX_SAFE_POINTS
    else:
        threshold = MAX_TERNARY_GRID_POINTS if is_advanced else MAX_SAFE_TERNARY_POINTS
        
    if n_points > threshold:
        st.error(f"**Grid Size Too Large!**\n\n"
                 f"Your current settings will generate approximately **{n_points:,} points** "
                 f"(~{mem_mb:.1f} MB core memory), which exceeds the safe limit of **{threshold:,}**.\n\n"
                 f"**How to fix:**\n"
                 f"- Increase the grid step size (`Δϕ`)\n"
                 f"- Decrease the maximum volume fraction (`ϕ max`)\n"
                 f"- If you really need this resolution, enable 'Advanced / High-Resolution Grid' in the sidebar.")
        st.stop()
        
    return n_points, mem_mb

def check_ternary_composition(phi2_max: float, phi3_max: float):
    """Ensure that the maximum volume fractions do not leave zero room for solvent."""
    total_phi = phi2_max + phi3_max
    if total_phi >= 0.99:
        st.error(f"**Invalid Composition!**\n\n"
                 f"The maximum volume fractions for cosolute 2 and 3 sum up to {total_phi:.2f}. "
                 f"This leaves less than 1% for the solvent, which is unphysical and may cause numerical instability.\n\n"
                 f"**How to fix:**\n"
                 f"- Reduce `ϕ₂ max` or `ϕ₃ max` so their sum is strictly less than 0.99.")
        st.stop()

def validate_grid_request(model_type: str, settings: dict) -> tuple[bool, list[str], dict]:
    """Validate a grid request before creating a model or running a heavy simulation."""
    errors = []
    points = 0
    
    if model_type == "binary":
        points = estimate_binary_grid_points(
            settings.get("phiC_min", 0.0), 
            settings.get("phiC_max", 0.3), 
            settings.get("dphiC", 0.01)
        )
        if points > MAX_BINARY_GRID_POINTS:
            errors.append(f"Binary grid points ({points:,}) exceed limit ({MAX_BINARY_GRID_POINTS:,}).")
            
    elif model_type == "ternary":
        points = estimate_ternary_grid_points(
            settings.get("phi2_min", 0.0), settings.get("phi2_max", 0.3), settings.get("dphi2", 0.01),
            settings.get("phi3_min", 0.0), settings.get("phi3_max", 0.3), settings.get("dphi3", 0.01)
        )
        if points > MAX_TERNARY_GRID_POINTS:
            errors.append(f"Ternary grid points ({points:,}) exceed limit ({MAX_TERNARY_GRID_POINTS:,}).")
        
        # Check composition
        total_phi = settings.get("phi2_max", 0.3) + settings.get("phi3_max", 0.3)
        if total_phi >= 0.99:
             errors.append(f"Maximum volume fractions (ϕ2 + ϕ3 = {total_phi:.2f}) exceed physical limit of < 0.99.")
            
    else:
        errors.append(f"Unknown model_type: {model_type}")
        
    return len(errors) == 0, errors, {"points": points}

def validate_action_preconditions(action_name: str, session_state: dict) -> tuple[bool, list[str]]:
    """Check if the session state satisfies preconditions for a specific action."""
    errors = []
    
    if action_name == "fit":
        if not session_state.get("exp_data_loaded", False):
            errors.append("No valid experimental data uploaded for fitting.")
            
    elif action_name == "simulate_fitted":
        has_bin_fit = session_state.get("fitted_eps") is not None or session_state.get("fitted_epsTS") is not None
        has_tern_fit = session_state.get("fitted_eps2") is not None or session_state.get("fitted_eps3") is not None or session_state.get("fitted_epsTS2") is not None
        if not (has_bin_fit or has_tern_fit):
            errors.append("No fitted parameters found. Run parameter fitting first.")
            
    elif action_name == "generate_csv":
        if "results" not in session_state:
            errors.append("No simulation results available. Run a simulation first.")
            
    elif action_name == "plot":
        if "results" not in session_state:
            errors.append("No simulation results available to plot.")
            
    elif action_name == "ternary_fit":
        # Additional checks for ternary fitting if needed
        pass
        
    return len(errors) == 0, errors

def estimate_dataframe_size(rows: int, columns: int) -> dict:
    """Estimate the size of a dataframe to be downloaded."""
    # Assume 8 bytes per cell
    mem_bytes = rows * columns * 8
    mem_mb = mem_bytes / (1024 * 1024)
    return {"mem_mb": mem_mb, "rows": rows, "columns": columns, "is_safe": rows <= MAX_DOWNLOAD_ROWS}

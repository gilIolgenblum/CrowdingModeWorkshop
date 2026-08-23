import json
from datetime import datetime
import numpy as np
import streamlit as st
from validation import (
    estimate_binary_grid_points, 
    estimate_ternary_grid_points, 
    MAX_BINARY_GRID_POINTS, 
    MAX_TERNARY_GRID_POINTS
)

APP_VERSION = "0.1.0"

def _to_serializable(val):
    """Convert numpy arrays, nan values, and numeric types to standard JSON-serializable types."""
    if isinstance(val, np.ndarray):
        return [None if np.isnan(x) else x for x in val.tolist()]
    elif isinstance(val, (list, tuple)):
        return [None if (isinstance(x, float) and np.isnan(x)) else x for x in val]
    elif isinstance(val, float) and np.isnan(val):
        return None
    elif isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val

def serialize_session_state() -> str:
    """Serialize relevant streamlit session state variables to a JSON string."""
    state = st.session_state
    
    # Extract protein parameters
    protein = {
        "SASA": state.get("SASA", 419.0)
    }
    
    # Extract temperature
    temperature = {
        "use_T": state.get("use_T", True),
        "T": state.get("T_input", 298.15)
    }
    
    # Extract binary parameters
    binary_params = {
        "cosolute_preset": state.get("bin_cosolute_select", "Custom"),
        "nu": state.get("bin_nu", 1.0),
        "chi": state.get("bin_chi", 0.1),
        "chiTS": state.get("bin_chiTS", -0.05),
        "eps": state.get("bin_eps_input", 0.0),
        "epsTS": state.get("bin_epsts_input", 0.0),
        "dphiC": state.get("bin_dphiC", 0.001),
        "phiC_max": state.get("bin_phiC_max", 0.15)
    }
    
    # Extract ternary parameters
    ternary_params = {
        "pair_preset": state.get("tern_pair_select", "Custom"),
        "cosolute2_preset": state.get("tern_cosolute2_select", "Custom"),
        "cosolute3_preset": state.get("tern_cosolute3_select", "Custom"),
        "nu2": state.get("nu2", 1.0),
        "chi12": state.get("chi12", 0.1),
        "chiTS12": state.get("chiTS12", -0.05),
        "eps2": state.get("tern_eps2_input", 0.0),
        "epsTS2": state.get("tern_epsts2_input", 0.0),
        "nu3": state.get("nu3", 1.0),
        "chi13": state.get("chi13", 0.1),
        "chiTS13": state.get("chiTS13", -0.05),
        "eps3": state.get("tern_eps3_input", 0.0),
        "epsTS3": state.get("tern_epsts3_input", 0.0),
        "chi23": state.get("chi23", 0.0),
        "chiTS23": state.get("chiTS23", 0.0),
        "eps23": state.get("eps23", 0.0),
        "epsTS23": state.get("epsTS23", 0.0),
        "dphi2": state.get("tern_dphi2", 0.001),
        "dphi3": state.get("tern_dphi3", 0.001),
        "phi2_max": state.get("tern_phi2_max", 0.2),
        "phi3_max": state.get("tern_phi3_max", 0.2)
    }
    
    # Extract unit settings
    unit_settings = {
        "uploaded_conc_unit": state.get("uploaded_conc_unit", "molal"),
        "uploaded_energy_unit": state.get("uploaded_energy_unit", "kJ/mol")
    }
    
    # Extract upload modes
    upload_modes = {
        "bin_upload_mode": state.get("bin_upload_mode", "Single CSV File (concentration, dG, dH, TdS)"),
        "tern_upload_mode": state.get("tern_upload_mode", "Columns (conc2, conc3, potential)")
    }
    
    # Extract fitted parameters
    fitted_parameters = {
        "fitted_eps": state.get("fitted_eps"),
        "fitted_epsTS": state.get("fitted_epsTS"),
        "fitted_eps2": state.get("fitted_eps2"),
        "fitted_eps3": state.get("fitted_eps3"),
        "fitted_epsTS2": state.get("fitted_epsTS2"),
        "fitted_epsTS3": state.get("fitted_epsTS3")
    }
    
    # Extract fit diagnostics if available in the current model in session state
    fit_diagnostics = {
        "res": None,
        "resTS": None
    }
    solved_model = state.get("solved_model")
    if solved_model is not None:
        if hasattr(solved_model, "res") and solved_model.res is not None:
            fit_diagnostics["res"] = {
                "success": bool(getattr(solved_model.res, "success", False)),
                "message": str(getattr(solved_model.res, "message", ""))
            }
        if hasattr(solved_model, "resTS") and solved_model.resTS is not None:
            fit_diagnostics["resTS"] = {
                "success": bool(getattr(solved_model.resTS, "success", False)),
                "message": str(getattr(solved_model.resTS, "message", ""))
            }
            
    # Extract plot selections
    plot_selections = {
        "bin_plot_conc": state.get("bin_plot_conc", "phi"),
        "bin_plot_unit": state.get("bin_plot_unit", "kJ/mol"),
        "preset_plot": state.get("preset_plot", "ddG (3x3 contour)"),
        "custom_plot_option": state.get("custom_plot_option", "Standard Preset Plot")
    }
    
    # Extract experimental data
    experimental_data = {
        "exp_data_loaded": state.get("exp_data_loaded", False),
        "bin_sample_select": state.get("bin_sample_select", "None"),
        "tern_sample_select": state.get("tern_sample_select", "None"),
        "data": {
            "exp_conc_G": _to_serializable(state.get("exp_conc_G")),
            "exp_ddG": _to_serializable(state.get("exp_ddG")),
            "err_ddG": _to_serializable(state.get("err_ddG")),
            "exp_conc_T": _to_serializable(state.get("exp_conc_T")),
            "exp_ddH": _to_serializable(state.get("exp_ddH")),
            "err_ddH": _to_serializable(state.get("err_ddH")),
            "exp_TddS": _to_serializable(state.get("exp_TddS")),
            "err_TddS": _to_serializable(state.get("err_TddS")),
            
            "exp_conc2": _to_serializable(state.get("exp_conc2")),
            "exp_conc3": _to_serializable(state.get("exp_conc3")),
            "exp_conc2_G": _to_serializable(state.get("exp_conc2_G")),
            "exp_conc3_G": _to_serializable(state.get("exp_conc3_G")),
            "exp_conc2_T": _to_serializable(state.get("exp_conc2_T")),
            "exp_conc3_T": _to_serializable(state.get("exp_conc3_T")),
            "exp_val_G": _to_serializable(state.get("exp_val_G")),
            "exp_val_H": _to_serializable(state.get("exp_val_H")),
            "exp_val_S": _to_serializable(state.get("exp_val_S")),
            
            "raw_exp_ddG": _to_serializable(state.get("raw_exp_ddG")),
            "raw_err_ddG": _to_serializable(state.get("raw_err_ddG")),
            "raw_exp_ddH": _to_serializable(state.get("raw_exp_ddH")),
            "raw_err_ddH": _to_serializable(state.get("raw_err_ddH")),
            "raw_exp_TddS": _to_serializable(state.get("raw_exp_TddS")),
            "raw_err_TddS": _to_serializable(state.get("raw_err_TddS")),
            
            "raw_exp_val_G": _to_serializable(state.get("raw_exp_val_G")),
            "raw_exp_val_H": _to_serializable(state.get("raw_exp_val_H")),
            "raw_exp_val_S": _to_serializable(state.get("raw_exp_val_S"))
        }
    }
    
    payload = {
        "app_version": APP_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_type": state.get("last_model_type", "Binary Crowding Model"),
        "protein": protein,
        "temperature": temperature,
        "binary_params": binary_params,
        "ternary_params": ternary_params,
        "unit_settings": unit_settings,
        "upload_modes": upload_modes,
        "fitted_parameters": fitted_parameters,
        "fit_diagnostics": fit_diagnostics,
        "plot_selections": plot_selections,
        "experimental_data": experimental_data
    }
    
    return json.dumps(payload, indent=2)

def deserialize_session_file(file_content: str) -> dict:
    """Deserialize JSON session file content safely."""
    return json.loads(file_content)

def validate_session_payload(payload: dict) -> tuple[bool, str]:
    """Validate loaded session payload structure, values, and types."""
    required_sections = ["app_version", "model_type", "protein", "temperature", "unit_settings", "fitted_parameters", "experimental_data"]
    for section in required_sections:
        if section not in payload:
            return False, f"Missing required section: {section}"
            
    # Validate model type
    model_type = payload.get("model_type")
    if model_type not in ["Binary Crowding Model"]:
        return False, f"Invalid model_type: {model_type}"
        
    # Validate numeric parameters in protein
    protein = payload.get("protein", {})
    if not isinstance(protein.get("SASA"), (int, float)) or protein.get("SASA") <= 0:
        return False, "Protein SASA must be a positive number."
        
    # Validate temperature
    temp = payload.get("temperature", {})
    if not isinstance(temp.get("T"), (int, float)) or temp.get("T") <= 0:
        return False, "Temperature must be a positive number."
        
    # Validate concentration units
    unit_settings = payload.get("unit_settings", {})
    if unit_settings.get("uploaded_conc_unit") not in ["phi", "molar", "molal"]:
        return False, f"Invalid concentration unit: {unit_settings.get('uploaded_conc_unit')}"
    if unit_settings.get("uploaded_energy_unit") not in ["kcal/mol", "kJ/mol"]:
        return False, f"Invalid energy unit: {unit_settings.get('uploaded_energy_unit')}"
        
    # Validate binary params if binary model
    if model_type == "Binary Crowding Model":
        bin_params = payload.get("binary_params", {})
        for num_key in ["nu", "chi", "chiTS", "eps", "epsTS"]:
            if num_key not in bin_params or not isinstance(bin_params[num_key], (int, float)):
                return False, f"Missing or invalid binary parameter: {num_key}"
        if bin_params.get("nu") <= 0:
            return False, "Binary cosolute nu must be positive."
        if bin_params.get("dphiC") is not None and (not isinstance(bin_params["dphiC"], (int, float)) or bin_params["dphiC"] <= 0):
            return False, "Binary grid step size dphiC must be a positive number."
            
        pts = estimate_binary_grid_points(0.0, bin_params.get("phiC_max", 0.15), bin_params.get("dphiC", 0.001))
        if pts > MAX_BINARY_GRID_POINTS:
            return False, f"Restored binary grid points ({pts:,}) exceed limit ({MAX_BINARY_GRID_POINTS:,})."
            
    # Validate ternary params if ternary model
    pass
            
    # Validate experimental data arrays
    exp_data = payload.get("experimental_data", {})
    data_dict = exp_data.get("data", {})
    for array_key, array_val in data_dict.items():
        if array_val is not None:
            if not isinstance(array_val, list):
                return False, f"Experimental data key {array_key} must be a list or null."
            for item in array_val:
                if item is not None and not isinstance(item, (int, float)):
                    return False, f"Experimental data key {array_key} must contain only numbers or nulls."
                    
    return True, "Valid session file."

def apply_session_payload(payload: dict) -> None:
    """Restore state variables into streamlit session state from the payload."""
    state = st.session_state
    
    # Apply model type
    model_type = payload.get("model_type")
    state["last_model_type"] = model_type
    
    # Apply protein
    protein = payload.get("protein", {})
    state["SASA"] = protein.get("SASA", 419.0)
    
    # Apply temperature
    temp = payload.get("temperature", {})
    state["use_T"] = temp.get("use_T", True)
    state["T_input"] = temp.get("T", 298.15)
    
    # Apply unit settings
    unit_settings = payload.get("unit_settings", {})
    state["uploaded_conc_unit"] = unit_settings.get("uploaded_conc_unit", "molal")
    state["uploaded_energy_unit"] = unit_settings.get("uploaded_energy_unit", "kJ/mol")
    
    # Apply upload modes
    upload_modes = payload.get("upload_modes", {})
    state["bin_upload_mode"] = upload_modes.get("bin_upload_mode", "Single CSV File (concentration, dG, dH, TdS)")
    state["tern_upload_mode"] = upload_modes.get("tern_upload_mode", "Columns (conc2, conc3, potential)")
    
    # Apply plot selections
    plot_selections = payload.get("plot_selections", {})
    state["bin_plot_conc"] = plot_selections.get("bin_plot_conc", "phi")
    state["bin_plot_unit"] = plot_selections.get("bin_plot_unit", "kJ/mol")
    state["preset_plot"] = plot_selections.get("preset_plot", "ddG (3x3 contour)")
    state["custom_plot_option"] = plot_selections.get("custom_plot_option", "Standard Preset Plot")
    
    # Apply binary parameters
    if model_type == "Binary Crowding Model":
        bin_params = payload.get("binary_params", {})
        state["bin_cosolute_select"] = bin_params.get("cosolute_preset", "Custom")
        state["bin_nu"] = bin_params.get("nu", 1.0)
        state["bin_chi"] = bin_params.get("chi", 0.1)
        state["bin_chiTS"] = bin_params.get("chiTS", -0.05)
        state["bin_eps_input"] = bin_params.get("eps", 0.0)
        state["bin_epsts_input"] = bin_params.get("epsTS", 0.0)
        state["bin_dphiC"] = bin_params.get("dphiC", 0.001)
        state["bin_phiC_max"] = bin_params.get("phiC_max", 0.15)
        
    # Apply ternary parameters
    else:
        pass
        
    # Apply fitted parameters
    fitted_parameters = payload.get("fitted_parameters", {})
    state["fitted_eps"] = fitted_parameters.get("fitted_eps")
    state["fitted_epsTS"] = fitted_parameters.get("fitted_epsTS")
    state["fitted_eps2"] = fitted_parameters.get("fitted_eps2")
    state["fitted_eps3"] = fitted_parameters.get("fitted_eps3")
    state["fitted_epsTS2"] = fitted_parameters.get("fitted_epsTS2")
    state["fitted_epsTS3"] = fitted_parameters.get("fitted_epsTS3")
    
    # Apply experimental data
    exp_data = payload.get("experimental_data", {})
    state["exp_data_loaded"] = exp_data.get("exp_data_loaded", False)
    state["bin_sample_select"] = exp_data.get("bin_sample_select", "None")
    state["tern_sample_select"] = exp_data.get("tern_sample_select", "None")
    
    # Initialize all experimental data keys to None to prevent stale session leakage
    keys_to_clear = [
        "exp_conc_G", "exp_ddG", "err_ddG",
        "exp_conc_T", "exp_ddH", "err_ddH",
        "exp_TddS", "err_TddS",
        "exp_conc2", "exp_conc3",
        "exp_conc2_G", "exp_conc3_G",
        "exp_conc2_T", "exp_conc3_T",
        "exp_val_G", "exp_val_H", "exp_val_S",
        "raw_exp_ddG", "raw_err_ddG",
        "raw_exp_ddH", "raw_err_ddH",
        "raw_exp_TddS", "raw_err_TddS",
        "raw_exp_val_G", "raw_exp_val_H", "raw_exp_val_S"
    ]
    for key in keys_to_clear:
        state[key] = None
        
    data_dict = exp_data.get("data", {})
    for array_key, array_val in data_dict.items():
        if array_val is not None:
            # Replace nulls back to NaN
            cleaned = [np.nan if x is None else x for x in array_val]
            state[array_key] = np.array(cleaned)
        else:
            state[array_key] = None

    # Backward compatibility fallback
    if state.get("exp_conc2") is not None:
        if state.get("exp_conc2_G") is None:
            val_g = state.get("exp_val_G")
            if val_g is None or len(val_g) == len(state["exp_conc2"]):
                state["exp_conc2_G"] = state["exp_conc2"].copy() if isinstance(state["exp_conc2"], np.ndarray) else state["exp_conc2"]
        if state.get("exp_conc2_T") is None:
            val_h = state.get("exp_val_H")
            val_s = state.get("exp_val_S")
            target_len = len(val_h) if val_h is not None else (len(val_s) if val_s is not None else None)
            if target_len is None or target_len == len(state["exp_conc2"]):
                state["exp_conc2_T"] = state["exp_conc2"].copy() if isinstance(state["exp_conc2"], np.ndarray) else state["exp_conc2"]
    if state.get("exp_conc3") is not None:
        if state.get("exp_conc3_G") is None:
            val_g = state.get("exp_val_G")
            if val_g is None or len(val_g) == len(state["exp_conc3"]):
                state["exp_conc3_G"] = state["exp_conc3"].copy() if isinstance(state["exp_conc3"], np.ndarray) else state["exp_conc3"]
        if state.get("exp_conc3_T") is None:
            val_h = state.get("exp_val_H")
            val_s = state.get("exp_val_S")
            target_len = len(val_h) if val_h is not None else (len(val_s) if val_s is not None else None)
            if target_len is None or target_len == len(state["exp_conc3"]):
                state["exp_conc3_T"] = state["exp_conc3"].copy() if isinstance(state["exp_conc3"], np.ndarray) else state["exp_conc3"]
            
    # Set run flag so the app solves equilibrium after rerun
    state["session_restored"] = True

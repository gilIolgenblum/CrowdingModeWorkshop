import pandas as pd
import numpy as np

def validate_uploaded_data(df: pd.DataFrame, expected_cols: list) -> tuple[bool, str]:
    """
    Validate uploaded data frame against expected columns and types.
    
    Args:
        df: Pandas DataFrame to validate
        expected_cols: List of expected column names
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if df is None or df.empty:
        return False, "Uploaded file is empty or missing data."
    
    if len(df) > 100000:
        return False, f"Uploaded file is too large (max 100,000 rows). Uploaded: {len(df)} rows."
        
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns. Expected: {expected_cols}. Detected columns: {list(df.columns)}"
    
    # Check for non-numeric data, NaNs, zero, or negative concentrations
    for col in expected_cols:
        df_col = pd.to_numeric(df[col], errors='coerce')
        
        if df_col.isna().all():
            return False, f"Column '{col}' contains entirely non-numeric data or is completely empty."
            
        if df_col.isna().any():
            return False, f"Column '{col}' contains missing (NaN) values."
            
        if 'conc' in col.lower():
            if (df_col < 0).any():
                return False, f"Concentration column '{col}' contains negative values, which are unphysical."
                
    # Check for duplicate concentrations
    conc_cols = [col for col in expected_cols if 'conc' in col.lower()]
    if conc_cols:
        if df.duplicated(subset=conc_cols).any():
            return False, f"Duplicate concentration coordinates found in columns {conc_cols}."
            
    return True, "Data is valid."
    
def validate_ternary_coordinate_match(dfs: dict) -> tuple[bool, str]:
    """
    Validate that a set of ternary observable dataframes all share the same (conc2, conc3) coordinates.
    
    Args:
        dfs: dict mapping observable name to its DataFrame. e.g. {'dG': df_dg, 'dH': df_dh}
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not dfs:
        return False, "No dataframes provided."
        
    if len(dfs) == 1:
        return True, "Only one dataframe, coordinate match is trivial."
        
    base_name = list(dfs.keys())[0]
    base_df = dfs[base_name]
    
    if 'conc2' not in base_df.columns or 'conc3' not in base_df.columns:
        return False, f"Base dataframe '{base_name}' is missing 'conc2' or 'conc3'."
        
    base_coords = set(zip(base_df['conc2'], base_df['conc3']))
    
    for name, df in list(dfs.items())[1:]:
        if 'conc2' not in df.columns or 'conc3' not in df.columns:
            return False, f"Dataframe '{name}' is missing 'conc2' or 'conc3'."
            
        coords = set(zip(df['conc2'], df['conc3']))
        
        if coords != base_coords:
            return False, f"Coordinate mismatch. The (conc2, conc3) pairs in '{name}' do not match those in '{base_name}'."
            
    return True, "Coordinates match."

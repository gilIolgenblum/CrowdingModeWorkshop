import numpy as np
import pandas as pd
from typing import Callable

def monte_carlo_subsampling(
    n_points: int,
    fit_and_extract: Callable[[np.ndarray], dict],
    n_iterations: int = 100,
    subsample_fraction: float = 0.8,
    progress_bar: bool = True
) -> pd.DataFrame:
    """
    Perform Monte Carlo cross-validation by repeatedly subsampling data and extracting fitted parameters.
    
    Args:
        n_points: The total number of data points available in the full dataset.
        fit_and_extract: A callable that accepts a 1D numpy array of randomly chosen integer indices. 
                         It must slice the data, perform the fit, and return a dictionary of fitted parameters.
        n_iterations: The number of times to perform the subsampling and fitting process.
        subsample_fraction: The fraction of data points to keep in each iteration (default 80%).
        progress_bar: If True, displays a tqdm progress bar.
        
    Returns:
        A pandas DataFrame where each row corresponds to one iteration, and columns are the extracted parameters.
    """
    n_sub = int(np.round(n_points * subsample_fraction))
    
    if n_sub < 1:
        raise ValueError("Subsample fraction is too small; must select at least 1 data point.")
        
    iterator = range(n_iterations)
    if progress_bar:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Monte Carlo Subsampling", total=n_iterations)
        except ImportError:
            pass
            
    results = []
    
    for _ in iterator:
        # Randomly choose indices without replacement
        idx = np.random.choice(n_points, size=n_sub, replace=False)
        
        # Call the user's closure
        params = fit_and_extract(idx)
        results.append(params)
        
    return pd.DataFrame(results)

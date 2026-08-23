import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union
from .constants import Constants

class Protein(Constants):
    '''
    Protein class, contains class variable and methods that depend on the protein propeties.

    Args:
        SASA: Change in solvent accesible surface area due to protein folding
    '''
    
    def __init__(self, SASA: float):
        self.SASA = SASA        
    def __str__(self):
        return f"Protein (SASA={self.SASA})"

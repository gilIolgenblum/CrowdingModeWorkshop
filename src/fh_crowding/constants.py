import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union

class Constants:
    '''
    Base class with shared constants and small numeric helpers.
    '''
    R: float = 8.314  # J/(mol·K)
    Vs: float = 0.018  # solvent molar vol in L/mol
    _EPS: float = 1e-12  # small epsilon for stable logs

    def cal_phiC(self, phi_min: float, phi_max: float, dphi: float) -> np.ndarray:
        '''Return an array of cosolute volume fractions.'''
        return np.arange(phi_min, phi_max, dphi)

    def _clip_phi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''Clip volume fractions to (EPS, 1-EPS) to avoid log singularities.'''
        return np.clip(x, self._EPS, 1 - self._EPS)

    def _log(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''Numerically-stable natural log for probabilities/volume fractions.'''
        return np.log(self._clip_phi(x))

    def _to_kj(self, x: Union[float, np.ndarray], T: Optional[float] = None) -> Union[float, np.ndarray]:
        T_use = T if T is not None else getattr(self, 'T', 298.0)
        return x * self.R * T_use / 1000.0

    def _to_kcal(self, x: Union[float, np.ndarray], T: Optional[float] = None) -> Union[float, np.ndarray]:
        return self._to_kj(x, T) / 4.184


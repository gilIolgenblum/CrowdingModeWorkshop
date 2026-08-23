import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union
from scipy.optimize import fsolve
from scipy.optimize import minimize
from .protein import Protein
from .cosolute import Cosolute

class CrowdingModel(Cosolute):
    '''
    Mean-field model class, contains class variable and methods that used to solve the folding thermodynamics of
    a protein-cosolute pair  for a set of SASA, nu, chi, and eps.

    
    Args:
        protien: A protein class - used to get protein parameters (SASA)
        cosol: A cosolute class - used to get cosolute parameters (nu, chi, chiTS)
        eps: soft interaction parameter
        epsTs: The entropic contribution of eps
        phiC_min: Minimal concetration (volume fraction)
        phiC_max: Maximal concetration (volume fraction)
        dphiC: Concetraion step (volume fraction)  
        T: Temperature (Kelvin)
    '''
    
    def __init__(self, protein: Protein, cosolute: Cosolute, eps: float = 0, epsTS: float = 0,
                 phiC_min: float = 0.0001, phiC_max: float = 0.15, dphiC: float = 0.0001, T: float = 298):
        # set variables
        self.T=T
        self.nu, self.chi, self.chiTS = cosolute.nu, cosolute.chi, cosolute.chiTS
        self.eps, self.epsTS, self.SASA = eps, epsTS, protein.SASA
        self.chiH, self.epsH = self.chi + self.chiTS, self.eps + self.epsTS
        self.a = self.nu**(1/3)
        self.phiC = self.cal_phiC(phiC_min, phiC_max, dphiC)
        self.phiS = 1-self.phiC
        self.molar = self.phiC / (self.nu * self.Vs)
        self.molal = self.phiC / (18 * self.phiS * self.nu)*1000
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.osm = self.cal_osm()
        
        # initialize empty arrays and variables
        self.phiCsurf, self.phiSsurf = np.zeros(self.phiC.shape), np.zeros(self.phiC.shape) 
        self.flag=False # Enable calculation of thermodynamic potential only after solving the equilibrium condition
        
        self.gamma, self.gammaC = 0, 0
        self.ddA, self.ddA_nu, self.ddA_chi, self.ddA_eps = 0, 0, 0, 0
        self.ddE, self.ddE_chi, self.ddE_eps = 0, 0 ,0
        self.TddS, self.TddS_nu, self.TddS_chi, self.TddS_eps = 0, 0, 0 ,0
        
        ## _Ms = per protein domain volume "Ms"
        self.gamma_Ms, self.gammaC_Ms = 0, 0
        self.ddA_Ms, self.ddA_nu_Ms, self.ddA_chi_Ms, self.ddA_eps_Ms = 0, 0, 0, 0
        self.ddE_Ms, self.ddE_chi_Ms, self.ddE_eps_Ms = 0, 0 ,0
        self.TddS_Ms, self.TddS_nu_Ms, self.TddS_chi_Ms, self.TddS_eps_Ms = 0, 0, 0 ,0
        self.dddA_Ms, self.dddE_Ms, self.TdddS_Ms = 0, 0, 0

        ## _kj = in units of kilo joul 
        self.ddA_kj, self.ddA_nu_kj, self.ddA_chi_kj, self.ddA_eps_kj = 0, 0, 0, 0
        self.ddE_kj, self.ddE_chi_kj, self.ddE_eps_kj = 0, 0 ,0
        self.TddS_kj, self.TddS_nu_kj, self.TddS_chi_kj, self.TddS_eps_kj = 0, 0, 0 ,0
        
        ## _kcal = in units of kilo cal, for unfolding procees 
        self.ddA_kcal, self.ddA_nu_kcal, self.ddA_chi_kcal, self.ddA_eps_kcal = 0, 0, 0, 0
        self.ddE_kcal, self.ddE_chi_kcal, self.ddE_eps_kcal = 0, 0 ,0
        self.TddS_kcal, self.TddS_nu_kcal, self.TddS_chi_kcal, self.TddS_eps_kcal = 0, 0, 0 ,0
        
    def __str__(self):
        return f"Crowding Mean-Field Model Parameters:\nSoft Interactions (\u03B5={self.eps}, \u03B5ₛ={self.epsTS}) \nProtein (SASA={self.SASA}) \nCosolute (\u03BD={self.nu}, \u03C7={self.chi}, \u03C7ₛ={self.chiTS})"
 
    def fit_eps(self, exp_conc: Sequence[float], exp_ddG: Sequence[float], concentration_type: str = 'phi', disp: bool = True, progress_bar: bool = False, auto_solve: bool = True):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameter, eps

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddG = np.array(exp_ddG)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            K = exp_conc * 18 * self.nu / 1000.0
            exp_conc = K / (1 + K)
        else:
            raise Exception("Concetration type can be either molar, molal, or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.flag=True
        
        tqdm_cb = None
        pbar = None
        if progress_bar:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(desc="Fitting eps", unit="iter")
                def tqdm_cb(xk): pbar.update(1)
            except ImportError:
                pass

        self.res = minimize(self.msd_fit_eps, self.eps, (exp_ddG,), options={'disp':disp}, callback=tqdm_cb)
        
        if pbar is not None:
            pbar.close()
            
        try:
            self.eps = float(self.res.x[0])
        except Exception:
            self.eps = float(self.res.x)
        # return to model arrays
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        # Note: We intentionally do not solve the full grid automatically here 
        # to prevent out-of-memory errors on large UI grids. 
        # The user must call `solve_equil()` explicitly if they want the full simulation.
        if auto_solve:
            self.solve_equil()

    def fit_epsTS(self, exp_conc: Sequence[float], exp_ddH: Sequence[float], exp_TddS: Sequence[float],
                  concentration_type: str = 'phi', disp: bool = True, progress_bar: bool = False, auto_solve: bool = True):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameter, eps

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddH = np.array(exp_ddH)
        exp_TddS = np.array(exp_TddS)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            K = exp_conc * 18 * self.nu / 1000.0
            exp_conc = K / (1 + K)
        else:
            raise Exception("Concetration type can be either molar or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        
        tqdm_cb = None
        pbar = None
        if progress_bar:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(desc="Fitting epsTS", unit="iter")
                def tqdm_cb(xk): pbar.update(1)
            except ImportError:
                pass

        self.resTS = minimize(self.msd_fit_epsTS, np.array(self.epsTS), (exp_ddH, exp_TddS), options={'disp':disp}, callback=tqdm_cb)
        
        if pbar is not None:
            pbar.close()
            
        try:
            self.epsTS = float(self.resTS.x[0])
        except Exception:
            self.epsTS = float(self.resTS.x)
        self.epsH = self.eps + self.epsTS
        # return to model arrays
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        # Note: We intentionally do not solve the full grid automatically here 
        # to prevent out-of-memory errors on large UI grids. 
        # The user must call `solve_equil()` explicitly if they want the full simulation.
        if auto_solve:
            self.solve_equil()
        
    def msd_fit_eps(self, eps: Union[float, np.ndarray], ddG: np.ndarray):
        self.eps = eps
        self.solve_equil()
        return ((ddG-self.ddA_kj)**2).sum()
        
    def msd_fit_epsTS(self, epsTS: Union[float, np.ndarray], ddH: np.ndarray, TddS: np.ndarray):
        self.epsTS=epsTS
        self.epsH = self.eps + self.epsTS
        self.solve_equil()
        return ((ddH-self.ddE_kj)**2).sum() + ((TddS-self.TddS_kj)**2).sum()  
    
    def fit_a(self, exp_conc: Sequence[float], exp_ddG: Sequence[float], concentration_type: str = 'phi', disp: bool = False, progress_bar: bool = False, auto_solve: bool = True):
        ''' 
        Fit the experimental folding free energy to resolve the protain domain size

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddG = np.array(exp_ddG)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            K = exp_conc * 18 * self.nu / 1000.0
            exp_conc = K / (1 + K)
        else:
            raise Exception("Concetration type can be either molar, molal, or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.flag=True
        self.a=np.zeros(self.phiC.shape)+self.nu**(1/3)
        
        tqdm_cb = None
        pbar = None
        if progress_bar:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(desc="Fitting a", unit="iter")
                def tqdm_cb(xk): pbar.update(1)
            except ImportError:
                pass

        bounds = [(1.0, None) for _ in self.a]
        self.res = minimize(self.msd_fit_a, self.a, (exp_ddG,), bounds=bounds, options={'disp':disp}, callback=tqdm_cb)
        
        if pbar is not None:
            pbar.close()
            
        self.a_vec=self.a
        self.fitted_a=self.a
        #print(self.a_vec)
        self.phiC_a=self.phiC
        if disp:
            plt.figure()
            plt.plot(self.phiC,self.a_vec,'o')
            plt.xlabel(r'$\phi_C$')
            plt.ylabel(r'$a$')
        # return to model arrays
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        if auto_solve:
            self.solve_equil()
        
    def msd_fit_a(self, a: np.ndarray, ddG: np.ndarray):
        self.a=a
        self.solve_equil()
        return ((ddG-self.ddA_kj)**2).sum()
    
    def condition(self, corr_phiCsurf: float, i: int):
        ''' 
        Equilibrium condition. Solved using SciPy's fsolve at a given concetration.
        
        Args:
            corr_phiCsurf: Corrent value of volume fraction in protein domain
            i: Concetration index
        '''
        corr_phiCmix = self.cal_phiCmix(corr_phiCsurf)
        corr_phiSmix = 1-corr_phiCmix
        corr_phiSsurf = 1-corr_phiCsurf
        
        if corr_phiCmix<0 or corr_phiSmix<0 or corr_phiSsurf<0:
            return 100
        if self.a <= 0:
            return 1
        
        return self.cal_muCsurf(corr_phiSsurf,corr_phiCmix,corr_phiSmix) - (self.nu)*self.cal_muSsurf(corr_phiCsurf,corr_phiCmix,corr_phiSmix) - self.muC[i] + (self.nu)*self.muS[i]
    def interpolate_a(self) -> np.ndarray:
        """
        Interpolate the domain size 'a' over the dense self.phiC grid, based on 
        the experimental points fitted in fit_a (stored in self.phiC_a and self.a_vec).
        Uses Pade[2,1] if enough points (>=4) are available, else linear interpolation.
        Anchors the curve at (phiC=0, a=nu**(1/3)).
        Clamps values outside the experimental range (no extrapolation).
        """
        if not hasattr(self, 'a_vec') or not hasattr(self, 'phiC_a'):
            # If not fitted yet, return the current self.a (scalar or array)
            if isinstance(self.a, np.ndarray) and len(self.a) == len(self.phiC):
                return self.a
            elif isinstance(self.a, (int, float)) or (isinstance(self.a, np.ndarray) and len(self.a) == 1):
                return np.full_like(self.phiC, float(np.mean(self.a)))
            else:
                return np.full_like(self.phiC, self.nu**(1/3))

        # Build training data, anchoring at phiC = 0
        x_data = np.concatenate(([0.0], self.phiC_a))
        y_data = np.concatenate(([self.nu**(1/3)], self.a_vec))

        # Sort just in case
        sort_idx = np.argsort(x_data)
        x_data = x_data[sort_idx]
        y_data = y_data[sort_idx]

        # Ensure uniqueness
        x_data, unique_idx = np.unique(x_data, return_index=True)
        y_data = y_data[unique_idx]

        a_interp = np.zeros_like(self.phiC)

        if len(x_data) >= 4:
            # Pade[2,1] interpolation/fit
            # p0 + p1*x + p2*x^2 - q1*(x*y) = y
            A_mat = np.vstack([np.ones_like(x_data), x_data, x_data**2, -x_data*y_data]).T
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, y_data, rcond=None)
            p0, p1, p2, q1 = coeffs
            
            def pade21(x):
                return (p0 + p1*x + p2*x**2) / (1 + q1*x)
            
            for i, x_val in enumerate(self.phiC):
                # Clamp to max experimental concentration to avoid extrapolation
                x_eval = min(x_val, x_data[-1])
                a_interp[i] = pade21(x_eval)
        else:
            # Linear interpolation
            from scipy.interpolate import interp1d
            f_lin = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value=(y_data[0], y_data[-1]))
            a_interp = f_lin(self.phiC)

        return a_interp

    def solve_equil(self, callback=None, progress_bar: bool = False):
        '''Solve the equilibrium condition across the concentration range.

        Uses rolling initial guesses to accelerate fsolve convergence.

        Args:
            callback: Optional callable ``(fraction: float) -> None`` invoked
                during the solve to report progress in [0, 1].  At most ~100
                calls are made regardless of grid size (one call per ~1 % of
                progress).  Pass ``None`` (default) to disable — notebooks are
                unaffected.
        '''
        n = len(self.phiC)
        report_every = max(1, n // 100)   # report at most ~100 times
        
        # Precompute the interpolated grid for 'a'
        a_grid = self.interpolate_a()
        
        A: list = []
        guess: Optional[float] = None
        
        iterator = range(n)
        if progress_bar:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="Solving equilibrium", total=n)
            except ImportError:
                pass

        for i in iterator:
            self.a = float(a_grid[i])
            if guess is None:
                guess = float(self.phiC[i])
            try:
                self.phiCsurf[i] = fsolve(self.condition, guess, args=(i,))[0]
            except Exception:
                self.phiCsurf[i] = fsolve(self.condition, float(self.phiC[i]), args=(i,))[0]
            A.append(self.a)
            guess = float(self.phiCsurf[i])
            if callback is not None and ((i + 1) % report_every == 0 or i == n - 1):
                callback((i + 1) / n)
        self.a = np.array(A)
        self.flag = True
        self.phiSsurf = 1 - self.phiCsurf
        self.cal_thermodynamics()
        
    def cal_thermodynamics(self):
        ''' Used after the equilbrium condition is solved. Calculates all model results including: preferential interaction coefficient, folding free energy, energy, and entropy.'''
        self.scaling = self.cal_scaling()
        self.phiCmix = self.cal_phiCmix()
        self.phiSmix = 1-self.phiCmix
        self.cal_gamma()
        self.cal_gammaC()
        self.cal_Free_Energy_nu()
        self.cal_Free_Energy_chi()
        self.cal_Free_Energy_eps()
        self.cal_Free_Energy()
        self.cal_Energy_chi()
        self.cal_Energy_eps()
        self.cal_Energy()
        self.cal_Entropy_nu()
        self.cal_Entropy_chi()
        self.cal_Entropy_eps()
        self.cal_Entropy()
        self.muCsurf = self.cal_muCsurf()
        self.muSsurf = self.cal_muSsurf()
        
    def cal_scaling(self, phiCsurf=-1):
        ''' 
        Calculate the scaling factor for a given volume fraction in the protein domain

        Arg:
            corr_phiCsurf: The current cosolute concentration in the protein domain
        
        Details:
            If used to solve the equilibrium condition, suply the current cosolute protein domain concetration to return the appropriate scaled volume.
            Otherwise returns the scaled volumes that corresond to the concetration array given by the instance variable. 
        '''
        
        if phiCsurf==-1:
            assert self.flag, 'Run solve_equil first'    
            phiCsurf=self.phiCsurf
        return 1-( 0.5*(1-1/self.a)*(1-phiCsurf) )
                  
    def cal_phiCmix(self, phiCsurf=-1):
        ''' 
        Calculate the scaled volume fraction in the protein domain for a given volume fraction

        Arg:
            phiCsurf: The cosolute concentration in the protein domain
        
        Details:
            If used to solve the equilibrium condition, suply the current cosolute protein domain concetration to return the appropriate scaled concetration.
            Otherwise returns the scaled concetrations that corresond to the concetration array given by the instance variable. 
        '''
        if phiCsurf==-1:
            assert self.flag, 'Run solve_equil first'  
            return self.phiCsurf / self.scaling 
        return phiCsurf / self.cal_scaling(phiCsurf)
         
    def cal_muCsurf(self, phiSsurf: Union[int, float, np.ndarray] = -1,
                    phiCmix: Union[int, float, np.ndarray] = -1,
                    phiSmix: Union[int, float, np.ndarray] = -1):
        ''' 
        Calculate the protein domain chemical potential of the cosolute

        Arg:
            phiSsurf : The solvent concentration in the protein domain
            phiCmix : The cosolute concentration in the mixing domain
            phiSmix : The solvent concetration in the mixing domain
            
        Details:
            If used to solve the equilibrium condition, suply the concetrations (see Arg) to return the chemical potential at the specific concetrations.
            Otherwise returns the chemical potentials for the concetration arrays corespnding to the instance variables. 
        '''
        
        if phiSsurf==-1 or phiCmix==-1 or phiSmix==-1:
            assert self.flag, 'Run solve_equil first'    
            phiSsurf, phiCmix, phiSmix= self.phiSsurf, self.phiCmix, self.phiSmix
        return self._log(phiCmix) + 1 - self.nu*phiSmix - phiCmix + self.chi*self.nu*phiSsurf**2 + self.eps*self.nu/self.a
        
    def cal_muSsurf(self, phiCsurf: Union[int, float, np.ndarray] = -1,
                    phiCmix: Union[int, float, np.ndarray] = -1,
                    phiSmix: Union[int, float, np.ndarray] = -1):
        ''' 
        Calculate the protein domain chemical potential of the cosolute

        Arg:
            phiCsurf : The cosolute concentration in the protein domain
            phiCmix : The cosolute concentration in the mixing domain
            phiSmix : The solvent concetration in the mixing domain
            
        Details:
            If used to solve the equilibrium condition, suply the concetrations (see Arg) to return the chemical potential at the specific concetrations.
            Otherwise returns the chemical potentials for the concetration arrays corespnding to the instance variables. 
        '''
        
        if phiCsurf==-1 or phiCmix==-1 or phiSmix==-1:
            assert self.flag, 'Run solve_equil first'    
            phiCsurf, phiCmix, phiSmix= self.phiCsurf, self.phiCmix, self.phiSmix
        return self._log(phiSmix) + 1 - phiSmix - phiCmix/self.nu + self.chi*phiCsurf**2
        
     
    def cal_Free_Energy_nu(self):
        ''' Calculate the contribution of excluded volume to the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        term1 = ((self.phiC-self.phiCsurf)) + ((1-self.phiCsurf))*self._log(1-self.phiC) 
        term2 = -self.scaling*((1-self.phiCmix))*self._log(1-self.phiCmix)
        term3 = ((self.phiCsurf-self.phiC)/self.nu) + (self.phiCsurf/self.nu)*self._log(self.phiC)
        term4 = -self.scaling*(self.phiCmix/self.nu)*self._log(self.phiCmix)
        self.ddA_nu_Ms = term1+term2+term3+term4
        self.ddA_nu = self.for_all_SASA(self.ddA_nu_Ms)
        self.ddA_nu_kj = self._to_kj(self.ddA_nu)
        self.ddA_nu_kcal = self._to_kcal(self.ddA_nu)
        
    def cal_Free_Energy_chi(self):
        ''' Calculate the contribution of non-ideal interactions to the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        term1 = self.chi*(self.phiCsurf-self.phiC)*(1-self.phiC)
        term2 = self.chi*self.phiC*(1-self.phiCsurf) 
        term3 = -self.chi*(1-self.phiCsurf)*self.phiCsurf
        self.ddA_chi_Ms = term1+term2+term3
        self.ddA_chi = self.for_all_SASA(self.ddA_chi_Ms)
        self.ddA_chi_kj = self._to_kj(self.ddA_chi)
        self.ddA_chi_kcal = self._to_kcal(self.ddA_chi)  

    def cal_Free_Energy_eps(self):
        ''' Calculate the contribution of soft interactions to the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddA_eps_Ms =  -(self.eps*self.phiCsurf)/(self.a)
        self.ddA_eps = self.for_all_SASA(self.ddA_eps_Ms)
        self.ddA_eps_kj = self._to_kj(self.ddA_eps)
        self.ddA_eps_kcal = self._to_kcal(self.ddA_eps)

    def cal_Free_Energy(self):
        ''' Calculate the folding free energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddA_Ms = self.ddA_nu_Ms + self.ddA_chi_Ms + self.ddA_eps_Ms
        self.ddA = self.ddA_nu + self.ddA_chi + self.ddA_eps
        self.ddA_kj = self.ddA_nu_kj + self.ddA_chi_kj + self.ddA_eps_kj
        self.ddA_kcal = self.ddA_kj/4.184

    def cal_Energy_chi(self):
        ''' Calculate the contribution of non-ideal mixing to the folding energy '''
        assert self.flag, 'Run solve_equil first'
        term1 = (self.chiH)*(self.phiCsurf-self.phiC)*(1-self.phiC) 
        term2 = (self.chiH)*self.phiC*(1-self.phiCsurf) 
        term3 = -(self.chiH)*(1-self.phiCsurf)*self.phiCsurf
        self.ddE_chi_Ms = term1+term2+term3
        self.ddE_chi = self.for_all_SASA(self.ddE_chi_Ms)
        self.ddE_chi_kj = self._to_kj(self.ddE_chi)
        self.ddE_chi_kcal = self._to_kcal(self.ddE_chi)
        
    def cal_Energy_eps(self):
        ''' Calculate the contribution of soft interactions to the folding energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddE_eps_Ms = -((self.epsH)*self.phiCsurf)/(self.a)
        self.ddE_eps = self.for_all_SASA(self.ddE_eps_Ms)
        self.ddE_eps_kj = self._to_kj(self.ddE_eps)
        self.ddE_eps_kcal = self._to_kcal(self.ddE_eps)

    def cal_Energy(self):
        ''' Calculate the folding energy '''
        assert self.flag, 'Run solve_equil first'
        self.ddE_Ms = self.ddE_chi_Ms + self.ddE_eps_Ms
        self.ddE = self.ddE_chi + self.ddE_eps
        self.ddE_kj = self.ddE_chi_kj + self.ddE_eps_kj
        self.ddE_kcal = self.ddE_kj/4.184

    def cal_Entropy_nu(self):
        ''' Calculate the contribution of excluded volume to the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_nu_Ms = -self.ddA_nu_Ms
        self.TddS_nu = -self.ddA_nu
        self.TddS_nu_kj = -self.ddA_nu_kj
        self.TddS_nu_kcal = self.TddS_nu_kj/4.184
        
    def cal_Entropy_chi(self):
        ''' Calculate the contribution of non-ideal mixing to the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_chi_Ms = self.ddE_chi_Ms-self.ddA_chi_Ms
        self.TddS_chi = self.ddE_chi-self.ddA_chi
        self.TddS_chi_kj = self.ddE_chi_kj-self.ddA_chi_kj
        self.TddS_chi_kcal = self.TddS_chi_kj/4.184
        
    def cal_Entropy_eps(self):
        ''' Calculate the contribution of soft-interactions to the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_eps_Ms = self.ddE_eps_Ms-self.ddA_eps_Ms
        self.TddS_eps = self.ddE_eps-self.ddA_eps
        self.TddS_eps_kj = self.ddE_eps_kj-self.ddA_eps_kj
        self.TddS_eps_kcal = self.TddS_eps_kj/4.184
        
    def cal_Entropy(self):
        ''' Calculate the folding entropy '''
        assert self.flag, 'Run solve_equil first'
        self.TddS_Ms = self.TddS_nu_Ms + self.TddS_chi_Ms + self.TddS_eps_Ms
        self.TddS = self.TddS_nu + self.TddS_chi + self.TddS_eps
        self.TddS_kj = self.TddS_nu_kj + self.TddS_chi_kj + self.TddS_eps_kj
        self.TddS_kcal = self.TddS_kj/4.184
    
    def cal_gamma(self):
        ''' Calculate the preferential hydration coefficient '''
        assert self.flag, 'Run solve_equil first'
        self.gamma_Ms = -(self.phiSsurf*(1 - (self.phiCsurf/self.phiC)*(self.phiS/self.phiSsurf) ))
        self.gamma = self.for_all_SASA(self.gamma_Ms)
        
    def cal_gammaC(self):
        ''' Calculate the preferential interactions coefficient '''
        assert self.flag, 'Run solve_equil first'
        self.gammaC_Ms = -(self.phiCsurf/self.nu*(1 - (self.phiSsurf/self.phiS)*(self.phiC/self.phiCsurf) ))
        self.gammaC = self.for_all_SASA(self.gammaC_Ms)
        
    def for_all_SASA(self, data):
        ''' 
        Convert potential from value per protein domain volume to value for the entire protein

        Arg:
            data: A thermodynamic potential: Free energy, energy, or entropy.
        '''
        return data*(self.SASA/30**(2/3))*self.a

    def to_pandas(self):
        self.results = pd.DataFrame({'phiC':self.phiC,
                            'phiCsurf':self.phiCsurf, 
                            'phiS':self.phiS, 
                            'phiSsurf':self.phiSsurf, 
                            'molar':self.molar,
                            'molal':self.molal,
                            'osm':self.osm,
                            'gamma_per_vol':self.gamma_Ms, 
                            'gamma':self.gamma,
                            'gammaC_per_vol':self.gammaC_Ms, 
                            'gammaC':self.gammaC,
                            'ddA_nu':self.ddA_nu, 
                            'ddA_chi':self.ddA_chi, 
                            'ddA_eps':self.ddA_eps, 
                            'ddA':self.ddA,
                            'ddE_chi':self.ddE_chi, 
                            'ddE_eps':self.ddE_eps, 
                            'ddE':self.ddE,
                            'TddS_nu':self.TddS_nu, 
                            'TddS_chi':self.TddS_chi, 
                            'TddS_eps':self.TddS_eps, 
                            'TddS':self.TddS,
                            'ddA_nu_kJ':self.ddA_nu_kj, 
                            'ddA_chi_kJ':self.ddA_chi_kj, 
                            'ddA_eps_kJ':self.ddA_eps_kj, 
                            'ddA_kJ':self.ddA_kj,
                            'ddE_chi_kJ':self.ddE_chi_kj, 
                            'ddE_eps_kJ':self.ddE_eps_kj, 
                            'ddE_kJ':self.ddE_kj,
                            'TddS_nu_kJ':self.TddS_nu_kj, 
                            'TddS_chi_kJ':self.TddS_chi_kj, 
                            'TddS_eps_kJ':self.TddS_eps_kj, 
                            'TddS_kJ':self.TddS_kj})
    
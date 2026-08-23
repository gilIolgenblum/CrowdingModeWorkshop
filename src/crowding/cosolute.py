import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union
from .constants import Constants

class Cosolute(Constants):
    '''
    Cosolute class, contains class variable and methods that depend on the cosolute propeties.

    Args:
        nu: FH excluded volume parameter
        chi: FH non-ideal interaction parameter
        chiTS: The entropic contribution of chi
        phiC_min: Minimal concetration (volume fraction)
        phiC_max: Maximal concetration (volume fraction)
        dphiC: Concetraion step (volume fraction)
    '''
    
    def __init__(self, nu: float, chi: float, chiTS: float,
                 phiC_min: float = 0.0001, phiC_max: float = 0.35, dphiC: float = 0.0001):
        # set variables
        self.nu, self.chi, self.chiTS, self.phiC_min, self.phiC_max, self.dphiC = nu, chi, chiTS, phiC_min, phiC_max, dphiC
        self.chiH = self.chi + self.chiTS
        self.phiC = self.cal_phiC(self.phiC_min, self.phiC_max, self.dphiC)
        self.phiS = 1-self.phiC
        self.molar = self.phiC / (self.nu * self.Vs)
        self.molal = self.phiC / (18 * self.phiS * self.nu)*1000
        self.dG, self.dH, self.dS = self.cal_FH_free_energy(), self.cal_FH_enthalpy(), self.cal_FH_entropy()
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.osm = self.cal_osm()
        
    def __str__(self):
        return f"Cosolute (\u03BD={self.nu}, \u03C7={self.chi}, \u03C7ₛ={self.chiTS})"

    def cal_muC(self) -> np.ndarray:
        ''' Calculate the chemical potential of the cosolute '''
        return (1-self.phiC)*(1-(self.nu)) + self._log(self.phiC) + self.chi*(self.nu)*((1-self.phiC)**2)

    def cal_muS(self) -> np.ndarray:
        ''' Calculate the chemical potential of the solvent '''
        return self._log(1-self.phiC) + self.phiC*(1-(1/self.nu)) + self.chi*((self.phiC)**2)

    def cal_FH_free_energy(self) -> np.ndarray:
        ''' Calculate the cosolute-solvent FH mixing free energy '''
        return self.phiS*self._log(self.phiS) + 1/self.nu*self.phiC*self._log(self.phiC) + self.chi*self.phiS*self.phiC

    def cal_FH_entropy(self) -> np.ndarray:
        ''' Calculate the cosolute-solvent FH mixing entropy '''
        return -self.phiS*self._log(self.phiS) - 1/self.nu*self.phiC*self._log(self.phiC) + self.chiTS*self.phiS*self.phiC
    
    def cal_FH_enthalpy(self) -> np.ndarray:
        ''' Calculate the cosolute-solvent FH mixing enthalpy '''
        return self.chiH*self.phiS*self.phiC

    def cal_osm(self) -> np.ndarray:
        ''' calculates the cosolute osmotic pressure'''
        return -(self.muS) / self.Vs


class CosoluteMixture(Constants):
    '''
    Cosolutes class, contains class variable and methods that depend on the properties of two cosolutes.

    Args:
        nu2 and nu3: FH excluded volume parameters
        chi12, chi13, and chi23: FH non-ideal interaction parameters
        chiTS12, chiTS13, and chiTS23: The entropic contribution of chi12, chi13, and chi23
        phi2_min: Minimal concetration (volume fraction) of species 2
        phi2_max: Maximal concetration (volume fraction) of species 2
        dphi2: Concetraion step (volume fraction) of species 2
        phi3_min: Minimal concetration (volume fraction) of species 3
        phi3_max: Maximal concetration (volume fraction) of species 3
        dphi3: Concetraion step (volume fraction) of species 3    '''
    
    def __init__(self, nu2: float, nu3: float, chi12: float, chi13: float, chi23: float,
                 chiTS12: float, chiTS13: float, chiTS23: float,
                 phi2_min: float = 0.0001, phi2_max: float = 0.35, dphi2: float = 0.0001,
                 phi3_min: float = 0.0001, phi3_max: float = 0.35, dphi3: float = 0.0001):
        self.nu2, self.nu3 = nu2, nu3 
        self.chi12, self.chi13, self.chi23 = chi12, chi13, chi23 
        self.chiTS12, self.chiTS13, self.chiTS23 = chiTS12, chiTS13, chiTS23
        self.phi2_min, self.phi2_max, self.dphi2 = phi2_min, phi2_max, dphi2
        self.phi3_min, self.phi3_max, self.dphi3 = phi3_min, phi3_max, dphi3
        # Enthalpic parts (corrected assignments)
        self.chiH12 = self.chi12 + self.chiTS12
        self.chiH13 = self.chi13 + self.chiTS13
        self.chiH23 = self.chi23 + self.chiTS23 
        phi2 = self.cal_phiC(self.phi2_min, self.phi2_max, self.dphi2)
        phi3 = self.cal_phiC(self.phi3_min, self.phi3_max, self.dphi3)
        self.phi2, self.phi3 = np.meshgrid(phi2,phi3)
        self.phi1 = 1-self.phi2-self.phi3
        self.phi1[self.phi1<0]=np.nan
        self.molar2 = self.phi2 / (self.nu2 * self.Vs)
        self.molar3 = self.phi3 / (self.nu3 * self.Vs)
        self.molal2 = self.phi2/self.phi1 / (18*self.nu2/1000)
        self.molal3 = self.phi3/self.phi1 / (18*self.nu3/1000)
        self.dGnonIdeal, self.TdSideal = self.cal_FH_nonideal_free_energy(), self.cal_FH_ideal_TS()
        self.dG_FH =  self.cal_FH_free_energy()
        self.mu1, self.mu2, self.mu3 = self.cal_mu1(), self.cal_mu2(), self.cal_mu3()
        self.osm = self.cal_osm()
        
    def __str__(self):
        return "Cosolutes:\n(\u03BD\N{SUBSCRIPT TWO}={}, \u03C7\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO}={}, \u03C7ₛ\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO}={})\n(\u03BD\N{SUBSCRIPT THREE}={}, \u03C7\N{SUBSCRIPT ONE}\N{SUBSCRIPT THREE}={}, \u03C7ₛ\N{SUBSCRIPT ONE}\N{SUBSCRIPT THREE}={})\n(\u03C7\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={}, \u03C7ₛ\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}={})".format(self.nu2,self.chi12,self.chiTS12,self.nu3,self.chi13,self.chiTS13,self.chi23,self.chiTS23)

    def cal_mu1(self):
        ''' calculates the solvent chemical potential in the bulk'''
        return self._log(self.phi1) + self.phi2*(1-1/self.nu2) + self.phi3*(1-1/self.nu3) + (1-self.phi1)* (self.chi12*self.phi2+self.chi13*self.phi3) - self.chi23*(self.phi2*self.phi3)/self.nu2
    
    def cal_mu2(self):
        ''' calculates cosolute #2 chemical potential in the bulk'''
        return self._log(self.phi2) + 1-self.nu2*self.phi1 - self.phi2 - self.nu2/self.nu3*self.phi3 + (1-self.phi2)*(self.chi12*self.nu2*self.phi1+self.chi23*self.phi3) - self.chi13*self.nu2*self.phi1*self.phi3

    def cal_mu3(self):
        ''' calculates cosolute #3 chemical potential in the bulk'''
        return self._log(self.phi3) + 1-self.nu3*self.phi1 -self.nu3/self.nu2*self.phi2 -self.phi3 + (1-self.phi3)*(self.chi13*self.nu3*self.phi1+self.chi23*self.nu3/self.nu2*self.phi2) - self.chi12*self.nu3*self.phi1*self.phi2    

    def cal_osm(self):
        ''' calculates the cosolutes osmotic pressure for ternary solution'''
        return -(self.mu1) / self.Vs

    def cal_FH_ideal_TS(self):
        ''' Calculate the cosolute-solvent FH mixing entropy '''
        return -(self.phi1*self._log(self.phi1) + 1/self.nu2*self.phi2*self._log(self.phi2) + 1/self.nu3*self.phi3*self._log(self.phi3))
    
    def cal_FH_nonideal_free_energy(self):
        ''' Calculate the cosolute-solvent FH mixing enthalpy '''
        return self.chi12*self.phi1*self.phi2 + self.chi13*self.phi1*self.phi3 + self.chi23/self.nu2*self.phi2*self.phi3

    def cal_FH_free_energy(self):
        ''' Calculate the cosolute-solvent FH mixing free energy '''
        return self.dGnonIdeal - self.TdSideal

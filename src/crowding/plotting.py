import numpy as np
import matplotlib.pyplot as plt
from .binary import CrowdingModel
from .ternary import TernaryCrowdingModel


def _is_valid_data(x, y):
    if x is None or y is None:
        return False
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    
    if len(x_arr) == 1:
        val = x_arr[0]
        if val is None or (isinstance(val, (int, float, np.number)) and np.isnan(val)):
            return False
    if len(y_arr) == 1:
        val = y_arr[0]
        if val is None or (isinstance(val, (int, float, np.number)) and np.isnan(val)):
            return False
            
    return len(x_arr) == len(y_arr) and len(x_arr) > 0


def _clean_yerr(yerr, y_len):
    if yerr is None:
        return None
    yerr_arr = np.atleast_1d(yerr)
    if len(yerr_arr) == 1:
        val = yerr_arr[0]
        if val is None or (isinstance(val, (int, float, np.number)) and np.isnan(val)):
            return None
        return val
    if len(yerr_arr) != y_len:
        return None
    if np.all([v is None or (isinstance(v, (int, float, np.number)) and np.isnan(v)) for v in yerr_arr]):
        return None
    return yerr_arr


class BinaryPlotter:
    def __init__(self, model: CrowdingModel):
        self.model = model

    def plot_results(self, concentration_type='phi', exp_conc=np.nan, exp_ddG=np.nan, err_ddG=np.nan,
                    exp_concT=np.nan, exp_ddH=np.nan, exp_TddS=np.nan, err_ddH=np.nan, err_TddS=np.nan,
                    folding=True, show_G=True, show_HTS=True):
        ''' 
        Plot model results 

        Arg:
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
            exp_conc: experimental concetration
            exp_ddG: experimental free energy
            err_ddG: experimental error in free energy
            exp_concT: experimental concetration for the enthalpy entropy data set
            exp_ddH: experimental enthalpy
            exp_TddS: experimental entropy
            err_ddH: experimental error in enthalpy
            err_TddS: experimental error in entropy
            folding: plot folding data in kJ (True), plot unfolding data in kcal.
        '''
        
        assert self.model.flag, 'Run solve_equil first'
        if not folding:
            if exp_ddG is not None and not (isinstance(exp_ddG, float) and np.isnan(exp_ddG)):
                exp_ddG = np.array(exp_ddG) / 4.184
            if err_ddG is not None and not (isinstance(err_ddG, float) and np.isnan(err_ddG)):
                err_ddG = np.array(err_ddG) / 4.184
            if exp_ddH is not None and not (isinstance(exp_ddH, float) and np.isnan(exp_ddH)):
                exp_ddH = np.array(exp_ddH) / 4.184
            if err_ddH is not None and not (isinstance(err_ddH, float) and np.isnan(err_ddH)):
                err_ddH = np.array(err_ddH) / 4.184
            if exp_TddS is not None and not (isinstance(exp_TddS, float) and np.isnan(exp_TddS)):
                exp_TddS = np.array(exp_TddS) / 4.184
            if err_TddS is not None and not (isinstance(err_TddS, float) and np.isnan(err_TddS)):
                err_TddS = np.array(err_TddS) / 4.184

        if concentration_type == 'phi':
            conc = self.model.phiC
            str_conc = r'$\phi_C$'
        elif concentration_type=='molar':
            conc = self.model.molar
            str_conc = 'molar'
        elif concentration_type=='molal':
            conc = self.model.molal
            str_conc = 'molal'

        if folding:
            ddA, ddA_nu, ddA_chi, ddA_eps = self.model.ddA_kj, self.model.ddA_nu_kj, self.model.ddA_chi_kj, self.model.ddA_eps_kj
            ddE, ddE_chi, ddE_eps = self.model.ddE_kj, self.model.ddE_chi_kj, self.model.ddE_eps_kj
            TddS, TddS_nu, TddS_chi, TddS_eps = self.model.TddS_kj, self.model.TddS_nu_kj, self.model.TddS_chi_kj, self.model.TddS_eps_kj
            units = '[kJ]'
        else:
            ddA, ddA_nu, ddA_chi, ddA_eps = self.model.ddA_kcal, self.model.ddA_nu_kcal, self.model.ddA_chi_kcal, self.model.ddA_eps_kcal
            ddE, ddE_chi, ddE_eps = self.model.ddE_kcal, self.model.ddE_chi_kcal, self.model.ddE_eps_kcal
            TddS, TddS_nu, TddS_chi, TddS_eps = self.model.TddS_kcal, self.model.TddS_nu_kcal, self.model.TddS_chi_kcal, self.model.TddS_eps_kcal
            units = '[kcal]'
            
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8), layout="constrained")
        axes[0,0].plot(conc, self.model.gamma)
        axes[0,0].set_xlabel(str_conc)
        axes[0,0].set_ylabel(r'$\Delta\Gamma_S$')

        axes[0,1].plot(conc, self.model.osm)
        axes[0,1].set_xlabel(str_conc)
        axes[0,1].set_ylabel(r'$\Pi (Osmolal)$')

        axes[0,2].plot(conc, self.model.phiCsurf)
        axes[0,2].set_xlabel(str_conc)
        axes[0,2].set_ylabel(r'$\phi_C^{surf}$')

        def _display_not_fitted(ax, xlabel, ylabel):
            ax.clear()
            ax.text(0.5, 0.5, 'Not Fitted', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks([])
            ax.set_yticks([])

        if show_G:
            axes[1,0].plot(conc, ddA)
            axes[1,0].plot(conc, ddA_nu)
            axes[1,0].plot(conc, ddA_chi)
            axes[1,0].plot(conc, ddA_eps)
            if _is_valid_data(exp_conc, exp_ddG):
                axes[1,0].errorbar(exp_conc, exp_ddG, yerr=_clean_yerr(err_ddG, len(exp_ddG)), marker='o', ls='', capsize=10, label='_nolegend_')
            axes[1,0].set_xlabel(str_conc)
            axes[1,0].set_ylabel(r'$\Delta\Delta G_i^{0}$ '+units)
            axes[1,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        else:
            _display_not_fitted(axes[1,0], str_conc, r'$\Delta\Delta G_i^{0}$ '+units)
        
        if show_HTS:
            axes[1,1].plot(conc, ddE)
            axes[1,1].plot(conc, ddE_chi)
            axes[1,1].plot(conc, ddE_eps)
            if _is_valid_data(exp_concT, exp_ddH):
                axes[1,1].errorbar(exp_concT, exp_ddH, yerr=_clean_yerr(err_ddH, len(exp_ddH)), marker='o', ls='', capsize=10, label='_nolegend_')
            axes[1,1].set_xlabel(str_conc)
            axes[1,1].set_ylabel(r'$\Delta\Delta H_i^{0}$ '+units)
            axes[1,1].legend(['tot',r'$\chi$',r'$\varepsilon$'])
    
            axes[1,2].plot(conc, TddS)
            axes[1,2].plot(conc, TddS_nu)
            axes[1,2].plot(conc, TddS_chi)
            axes[1,2].plot(conc, TddS_eps)
            if _is_valid_data(exp_concT, exp_TddS):
                axes[1,2].errorbar(exp_concT, exp_TddS, yerr=_clean_yerr(err_TddS, len(exp_TddS)), marker='o', ls='', capsize=10, label='_nolegend_')
            axes[1,2].set_xlabel(str_conc)
            axes[1,2].set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
            axes[1,2].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        else:
            _display_not_fitted(axes[1,1], str_conc, r'$\Delta\Delta H_i^{0}$ '+units)
            _display_not_fitted(axes[1,2], str_conc, r'$T\Delta\Delta S_i^{0}$ '+units)

        if show_G:
            axes[2,0].plot(self.model.osm, ddA)
            axes[2,0].plot(self.model.osm, ddA_nu)
            axes[2,0].plot(self.model.osm, ddA_chi)
            axes[2,0].plot(self.model.osm, ddA_eps)
            axes[2,0].set_xlabel(r'$\Pi (Osmolal)$')
            axes[2,0].set_ylabel(r'$\Delta\Delta G_i^{0}$ '+units)
            axes[2,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        else:
            _display_not_fitted(axes[2,0], r'$\Pi (Osmolal)$', r'$\Delta\Delta G_i^{0}$ '+units)

        if show_HTS:
            axes[2,1].plot([-max(abs(ddE)),max(abs(ddE))], [-max(abs(ddE)),max(abs(ddE))], color="darkgrey",label='_nolegend_') 
            axes[2,1].plot([-max(abs(ddE)),max(abs(ddE))], [max(abs(ddE)),-max(abs(ddE))], color="darkgrey",label='_nolegend_')
            axes[2,1].plot(ddE, TddS)
            axes[2,1].plot(np.zeros(TddS_nu.shape), TddS_nu)
            axes[2,1].plot(ddE_chi, TddS_chi)
            axes[2,1].plot(ddE_eps, TddS_eps)
            if _is_valid_data(exp_ddH, exp_TddS):
                axes[2,1].plot(exp_ddH, exp_TddS, 'o', label='_nolegend_')   
            axes[2,1].set_xlabel(r'$\Delta\Delta H_i^{0}$ '+units)
            axes[2,1].set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
            axes[2,1].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
    
    
            if max(abs(ddE_chi)) != 0:
                axes[2,2].set_xlim([-max(abs(ddE_chi)),max(abs(ddE_chi))])
            else:
                axes[2,2].set_xlim([-max(abs(TddS_chi)),max(abs(TddS_chi))])
            axes[2,2].set_ylim([-max(abs(TddS_chi)),max(abs(TddS_chi))])
            axes[2,2].plot([-max(abs(ddE)),max(abs(ddE))], [-max(abs(ddE)),max(abs(ddE))], color="darkgrey",label='_nolegend_') 
            axes[2,2].plot([-max(abs(ddE)),max(abs(ddE))], [max(abs(ddE)),-max(abs(ddE))], color="darkgrey",label='_nolegend_')
            axes[2,2].plot(ddE, TddS)
            axes[2,2].plot(np.zeros(TddS_nu.shape), TddS_nu)
            axes[2,2].plot(ddE_chi, TddS_chi)
            axes[2,2].plot(ddE_eps, TddS_eps)
            axes[2,2].set_xlabel(r'$\Delta\Delta H_i^{0}$ '+units)
            axes[2,2].set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
            axes[2,2].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
            axes[2,2].locator_params(axis='both', nbins=3)
        else:
            _display_not_fitted(axes[2,1], r'$\Delta\Delta H_i^{0}$ '+units, r'$T\Delta\Delta S_i^{0}$ '+units)
            _display_not_fitted(axes[2,2], r'$\Delta\Delta H_i^{0}$ '+units, r'$T\Delta\Delta S_i^{0}$ '+units)
            
        return fig


    def _get_conc_info(self, concentration_type):
        if concentration_type == 'phi':
            return self.model.phiC, r'$\phi_C$'
        elif concentration_type == 'molar':
            return self.model.molar, 'molar'
        elif concentration_type == 'molal':
            return self.model.molal, 'molal'
        raise ValueError("concentration_type must be 'phi', 'molar', or 'molal'")

    def _get_thermo_data(self, folding):
        if folding:
            return (self.model.ddA_kj, self.model.ddA_nu_kj, self.model.ddA_chi_kj, self.model.ddA_eps_kj,
                    self.model.ddE_kj, self.model.ddE_chi_kj, self.model.ddE_eps_kj,
                    self.model.TddS_kj, self.model.TddS_nu_kj, self.model.TddS_chi_kj, self.model.TddS_eps_kj,
                    '[kJ]')
        else:
            return (self.model.ddA_kcal, self.model.ddA_nu_kcal, self.model.ddA_chi_kcal, self.model.ddA_eps_kcal,
                    self.model.ddE_kcal, self.model.ddE_chi_kcal, self.model.ddE_eps_kcal,
                    self.model.TddS_kcal, self.model.TddS_nu_kcal, self.model.TddS_chi_kcal, self.model.TddS_eps_kcal,
                    '[kcal]')

    def plot_gamma(self, concentration_type='phi'):
        conc, str_conc = self._get_conc_info(concentration_type)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(conc, self.model.gamma)
        ax.set_xlabel(str_conc)
        ax.set_ylabel(r'$\Delta\Gamma_S$')
        return fig

    def plot_ddG(self, concentration_type='phi', folding=True, exp_conc=np.nan, exp_ddG=np.nan, err_ddG=np.nan):
        conc, str_conc = self._get_conc_info(concentration_type)
        ddA, ddA_nu, ddA_chi, ddA_eps, _, _, _, _, _, _, _, units = self._get_thermo_data(folding)
        
        if not folding and _is_valid_data(exp_conc, exp_ddG):
            exp_ddG = np.array(exp_ddG) / 4.184
            if err_ddG is not None and not (isinstance(err_ddG, float) and np.isnan(err_ddG)):
                err_ddG = np.array(err_ddG) / 4.184

        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(conc, ddA, label='tot')
        ax.plot(conc, ddA_nu, label=r'$\nu$')
        ax.plot(conc, ddA_chi, label=r'$\chi$')
        ax.plot(conc, ddA_eps, label=r'$\varepsilon$')
        if _is_valid_data(exp_conc, exp_ddG):
            ax.errorbar(exp_conc, exp_ddG, yerr=_clean_yerr(err_ddG, len(exp_ddG)), marker='o', ls='', capsize=5, label='_nolegend_')
        ax.set_xlabel(str_conc)
        ax.set_ylabel(r'$\Delta\Delta G_i^{0}$ '+units)
        ax.legend()
        return fig

    def plot_ddH(self, concentration_type='phi', folding=True, exp_conc=np.nan, exp_ddH=np.nan, err_ddH=np.nan):
        conc, str_conc = self._get_conc_info(concentration_type)
        _, _, _, _, ddE, ddE_chi, ddE_eps, _, _, _, _, units = self._get_thermo_data(folding)
        
        if not folding and _is_valid_data(exp_conc, exp_ddH):
            exp_ddH = np.array(exp_ddH) / 4.184
            if err_ddH is not None and not (isinstance(err_ddH, float) and np.isnan(err_ddH)):
                err_ddH = np.array(err_ddH) / 4.184

        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(conc, ddE, label='tot')
        ax.plot(conc, ddE_chi, label=r'$\chi$')
        ax.plot(conc, ddE_eps, label=r'$\varepsilon$')
        if _is_valid_data(exp_conc, exp_ddH):
            ax.errorbar(exp_conc, exp_ddH, yerr=_clean_yerr(err_ddH, len(exp_ddH)), marker='o', ls='', capsize=5, label='_nolegend_')
        ax.set_xlabel(str_conc)
        ax.set_ylabel(r'$\Delta\Delta H_i^{0}$ '+units)
        ax.legend()
        return fig

    def plot_TddS(self, concentration_type='phi', folding=True, exp_conc=np.nan, exp_TddS=np.nan, err_TddS=np.nan):
        conc, str_conc = self._get_conc_info(concentration_type)
        _, _, _, _, _, _, _, TddS, TddS_nu, TddS_chi, TddS_eps, units = self._get_thermo_data(folding)
        
        if not folding and _is_valid_data(exp_conc, exp_TddS):
            exp_TddS = np.array(exp_TddS) / 4.184
            if err_TddS is not None and not (isinstance(err_TddS, float) and np.isnan(err_TddS)):
                err_TddS = np.array(err_TddS) / 4.184

        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(conc, TddS, label='tot')
        ax.plot(conc, TddS_nu, label=r'$\nu$')
        ax.plot(conc, TddS_chi, label=r'$\chi$')
        ax.plot(conc, TddS_eps, label=r'$\varepsilon$')
        if _is_valid_data(exp_conc, exp_TddS):
            ax.errorbar(exp_conc, exp_TddS, yerr=_clean_yerr(err_TddS, len(exp_TddS)), marker='o', ls='', capsize=5, label='_nolegend_')
        ax.set_xlabel(str_conc)
        ax.set_ylabel(r'$T\Delta\Delta S_i^{0}$ '+units)
        ax.legend()
        return fig

    def plot_mu(self, concentration_type='phi'):
        conc, str_conc = self._get_conc_info(concentration_type)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(conc, self.model.muC, label=r'$\mu_C$')
        ax.set_xlabel(str_conc)
        ax.set_ylabel('Cosolute Chemical Potential')
        ax.legend()
        return fig

    def plot_osm(self, concentration_type='phi'):
        conc, str_conc = self._get_conc_info(concentration_type)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(conc, self.model.osm)
        ax.set_xlabel(str_conc)
        ax.set_ylabel(r'$\Pi$ (Osmolal)')
        return fig

    def plot_EEC(self, folding=True, exp_ddH=np.nan, exp_TddS=np.nan):
        _, _, _, _, ddE, ddE_chi, ddE_eps, TddS, TddS_nu, TddS_chi, TddS_eps, units = self._get_thermo_data(folding)
        
        if not folding:
            if _is_valid_data(exp_ddH, exp_TddS):
                exp_ddH = np.array(exp_ddH) / 4.184
                exp_TddS = np.array(exp_TddS) / 4.184
        
        fig, ax = plt.subplots(figsize=(5,5))
        max_val = max(np.max(np.abs(ddE)), np.max(np.abs(TddS)))
        ax.plot([-max_val, max_val], [-max_val, max_val], color="darkgrey", label='_nolegend_') 
        ax.plot([-max_val, max_val], [max_val, -max_val], color="darkgrey", label='_nolegend_')
        
        ax.plot(ddE, TddS, label='tot')
        ax.plot(np.zeros(TddS_nu.shape), TddS_nu, label=r'$\nu$')
        ax.plot(ddE_chi, TddS_chi, label=r'$\chi$')
        ax.plot(ddE_eps, TddS_eps, label=r'$\varepsilon$')
        
        if _is_valid_data(exp_ddH, exp_TddS):
            ax.plot(exp_ddH, exp_TddS, 'o', label='_nolegend_')
            
        ax.set_xlabel(r'$\Delta\Delta H_i^{0}$ ' + units)
        ax.set_ylabel(r'$T\Delta\Delta S_i^{0}$ ' + units)
        ax.legend()
        return fig


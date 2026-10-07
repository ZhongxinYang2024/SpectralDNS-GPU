import numpy as np
import torch
r'''
Notice:

Grid resolution:
k * eta_\max should be larger than 1, 2 is more desirable.

Time resolution:
CFL time step: Tc = \Delta x / u' \approx = 2 pi * \sqrt{3} / N
Kolmogorov time step: \tau_eta
\Delta t / Tc ~ (0.02, 0.1) 

same as T. Ishihara et al., JFM, 2007, Small-scale statistics in ....
'''

def get_config():
    config = {}
    
    ## grid parameter
    config['B'] = 1
    config['N'] = 256
    config['L'] = 2.0 * np.pi
    
    ## property parameter
    config['nu'] = 7. * 1e-4
    
    ## Time scheme
    config['TschemeName'] = 'RK4'
    config['dt'] = 1e-3
    config['Ttotal'] = 38.0

    
    ## Forced HIT
    config['IFforced'] = True #  False # 
    config['ForcedEnergy'] = 0.5 # 3/2 * u`**2
    config['ForceLowerAtKc'] = 2.5
    
    ## dealise method
    config['dealias_method'] = 'phase_shift' # '2_3 or 3_2' or 'phase-shift'
    config['device'] = 'cuda'
    
    ## save 
    config['savefolder'] = f"../data/N{config['N']}_nu{config['nu']*1000:.3f}m_T{config['Ttotal']}"
    config['savefolder_std'] = f"../data/N{config['N']}_nu{config['nu']*1000:.3f}m_T{config['Ttotal']}_Standard"
    config['Step_output'] = 50
    config['Step_save_begin'] = 10000  # after step_save_begin, then begin to save
    config['Step_save'] = 1000
    
    config['stat_id_begin'] = 0  # from which id, begin to get stat
    config['num_stat'] = 100


    return config

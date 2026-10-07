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

same as: Cao et al., 1999, POF, Statistics and structures of pressure in isotropic turbulence
'''


def get_config():
    config = {}
    
    ## grid paramete
    config['B'] = 1
    config['N'] = 32
    config['L'] = 2.0 * np.pi
    
    ## property parameter
    config['nu'] = 250. * 1e-4
    
    ## Time scheme
    config['TschemeName'] = 'RK4'
    config['dt'] =  8. * 1e-3
    config['Ttotal'] = 1.0 * 1e1
    
    ## Forced HIT
    config['IFforced'] = True
    config['ForcedEnergy'] = 0.5 # 3/2 * u`**2
    config['ForceLowerAtKc'] = 2.5
    
    ## dealise method
    config['dealias_method'] = 'phase_shift' # '2_3 or 3_2' or 'phase-shift'
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ## save 
    config['savefolder'] = f"../data/N{config['N']}_nu{config['nu']*1000:.3f}n_T{config['Ttotal']}"
    config['Step_output'] = 20
    config['Step_save'] = 500



    return config

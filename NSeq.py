import torch
from tools.FourierTransformation import rFFT3d_torch_user, rIFFT3d_torch_user
from tools.FourierTransformation import FFTfreq_torch_user, rFFTfreq_torch_user

def apply_projection(U_hat, KK, inv_K2_safe):
    # U_hat: [1, 3, ...]
    # k_dot_u = kx*ux + ky*uy + kz*uz
    k_dot_u = torch.sum(KK.unsqueeze(0) * U_hat, dim=1) # [B, N, N, N//2+1]
    
    # correction = k * (k dot u) / k^2
    correction = KK.unsqueeze(0) *  k_dot_u.unsqueeze(1) * inv_K2_safe.unsqueeze(0) 
    return U_hat - correction


def compute_nonlinearTerm(residual,  Uh, **kwargs):
    dealias_method = kwargs['dealias_method']
    KK = kwargs['KK'] # shape: [3, N, N, N//2 +1]
    inv_K2_safe = kwargs['inv_K2_safe'] # shape:[3, N, N, N//2 + 1]
        
    if dealias_method == '2_3':
        mask23 = kwargs['dealias_var']['mask23'] # shape [N, N, N // 2 +1]
        Uh_trunc = Uh * mask23 # dealias, significant
        
        ## omega = i k u
        cross_r = torch.cross(KK.unsqueeze(0), Uh_trunc.real, dim=-4)
        cross_i = torch.cross(KK.unsqueeze(0), Uh_trunc.imag, dim=-4)
        Omegah_trunc = -cross_i + 1j * cross_r
        
        U = rIFFT3d_torch_user(Uh_trunc)
        Omega = rIFFT3d_torch_user(Omegah_trunc)       
        
        # Lamb vector: w = u * omega
        W = torch.cross(U, Omega, dim = -4)     
        Wh = rFFT3d_torch_user(W)  # shape [1, 3, N, N, N //2 + 1]
        Wh *= mask23 
        
        
    elif dealias_method == 'phase_shift':
        mask_phase = kwargs['dealias_var']['mask_phase']
        shift_factor = kwargs['dealias_var']['shift_factor']
        unshift_factor = kwargs['dealias_var']['unshift_factor']
        
        # 1. use truncation ( kmax < sqrt(2)*N/3 )
        Uh_trunc = Uh * mask_phase.unsqueeze(0).unsqueeze(0) 
        
        # compute vorticity (Omega = i k x u)
        cross_r = torch.cross(KK.unsqueeze(0), Uh_trunc.real, dim=-4)
        cross_i = torch.cross(KK.unsqueeze(0), Uh_trunc.imag, dim=-4)
        Omegah_trunc = -cross_i + 1j * cross_r
        
        # ============================================
        # Grid 1:  (Standard Grid) 
        # ============================================
        U1 = rIFFT3d_torch_user(Uh_trunc)
        Omega1 = rIFFT3d_torch_user(Omegah_trunc)       
        W1 = torch.cross(U1, Omega1, dim=-4)     
        Wh1 = rFFT3d_torch_user(W1)
        
        # ============================================
        # Grid 2: (Shifted Grid)
        # ============================================
        # \times e^{i k \Delta} 
        Uh_shift = Uh_trunc * shift_factor.unsqueeze(0).unsqueeze(0)
        Omegah_shift = Omegah_trunc * shift_factor.unsqueeze(0).unsqueeze(0)
        
        U2 = rIFFT3d_torch_user(Uh_shift)
        Omega2 = rIFFT3d_torch_user(Omegah_shift)
        W2 = torch.cross(U2, Omega2, dim=-4)     
        Wh2_shifted = rFFT3d_torch_user(W2)
        
        # \times e^{-i k \Delta}
        Wh2 = Wh2_shifted * unshift_factor.unsqueeze(0).unsqueeze(0)
        

        #  (Phase averaging)
        Wh = 0.5 * (Wh1 + Wh2)
        
        Wh = Wh * mask_phase.unsqueeze(0).unsqueeze(0) 
        
        
    elif dealias_method == '3_2':
        B, _, _, _, _ = Uh.shape
        KKp = kwargs['dealias_var']['KKp']
        Np = kwargs['dealias_var']['Np']
        pad_start = kwargs['dealias_var']['pad_start']
        pad_end = kwargs['dealias_var']['pad_end']
        
        
        ## convective term
        Uh_pad = torch.zeros((B, 3, Np, Np, Np // 2 + 1), dtype = torch.complex64).to(Uh.device)
        ## padding # dealias, significant
        # Strict even-grid convention: exclude exact x/y/z Nyquist planes.
        Uh_pad[..., 0:pad_start, 0:pad_start, 0:pad_start] = Uh[..., 0:pad_start, 0:pad_start, 0:pad_start]
        Uh_pad[..., 0:pad_start, pad_end:   , 0:pad_start] = Uh[..., 0:pad_start, pad_end:   , 0:pad_start]
        Uh_pad[..., pad_end:   , 0:pad_start, 0:pad_start] = Uh[..., pad_end:   , 0:pad_start, 0:pad_start]
        Uh_pad[..., pad_end:   , pad_end:   , 0:pad_start] = Uh[..., pad_end:   , pad_end:   , 0:pad_start]
        
        # compute vorticity
        cross_r = torch.cross(KKp.unsqueeze(0), Uh_pad.real, dim=-4)
        cross_i = torch.cross(KKp.unsqueeze(0), Uh_pad.imag, dim=-4)
        Omegah_pad = -cross_i + 1j * cross_r
        
        U_pad = rIFFT3d_torch_user(Uh_pad)
        Omega_pad = rIFFT3d_torch_user(Omegah_pad)       
        
        W_pad = torch.cross(U_pad, Omega_pad, dim = -4)     
        Wh_pad = rFFT3d_torch_user(W_pad)  # shape [1, 3, Np, Np, Np //2 + 1]
        
        Wh = torch.zeros_like(Uh, device=Uh.device)
        Wh[..., 0:pad_start, 0:pad_start, 0:pad_start] = Wh_pad[..., 0:pad_start, 0:pad_start, 0:pad_start]
        Wh[..., 0:pad_start, pad_end:   , 0:pad_start] = Wh_pad[..., 0:pad_start, pad_end:   , 0:pad_start]
        Wh[..., pad_end:   , 0:pad_start, 0:pad_start] = Wh_pad[..., pad_end:   , 0:pad_start, 0:pad_start] 
        Wh[..., pad_end:   , pad_end:   , 0:pad_start] = Wh_pad[..., pad_end:   , pad_end:   , 0:pad_start]
        
    residual +=   (apply_projection(Wh.real, KK, inv_K2_safe) + 1j * apply_projection(Wh.imag, KK, inv_K2_safe))
    


class Equation_NS3d:
    def __init__(self):
        pass
    
    def compute_residual(self, Uh, boundary_conditions, **kwargs):
        '''
        Uh for easy to remember shape [1, 3, N, N, N // 2 +1]
        ''' 
        nu = kwargs['nu'] # viscosity
        K2 = kwargs['K2'] # shape: [3, N, N, N//2 +1]
        KK = kwargs['KK']
        inv_K2_safe = kwargs['inv_K2_safe']
        
        Uh = apply_projection(Uh.real, KK, inv_K2_safe) + 1j * apply_projection(Uh.imag, KK, inv_K2_safe)
    
        ## viscos term
        residual = - nu * K2.unsqueeze(0) * Uh
        
        ## unlinear term
        compute_nonlinearTerm(residual = residual,  Uh = Uh, **kwargs)

        ## force term
        if kwargs["IFforced"] == True:
            cc = kwargs["ForcedConst"]
            Kc = kwargs["ForceLowerAtKc"]
            
            condition = (K2 < Kc**2) # [N,N,N//2+1]

            cc_reshaped = cc.view(-1, 1, 1) 
            residual[:, :, condition] -= cc_reshaped * Uh[:, :, condition]
        
        ## force the mean to be  zero
        residual[..., 0, 0, 0] = 0.0 + 0.0j
        return residual    

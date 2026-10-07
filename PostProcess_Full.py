
import torch
import numpy as np
##
from tools.FourierTransformation import rIFFT3d_torch_user, rFFT3d_torch_user
from tools.MatrixOperator import TransferMatrix_alongAxis, Cal_Stat

class PostProcessor_FULL:
    def __init__(self, dt, N, K2, KK, nu, kmax, ll, IFforced=False, ForcedEnergy=None, ForceLowerAtKc=2.5):
        self.dt = dt
        self.N = N
        self.K2 = K2
        self.KK = KK
        self.nu = nu
        self.kmax = kmax
        self.ll = ll
        self.IFforced = IFforced
        self.ForcedEnergy = ForcedEnergy
        self.ForceLowerAtKc = ForceLowerAtKc
        
        ## history
        self.history =  {
            "tt":[], 
            "Ek": [],
            "kk": None,
            "Energy":[],
            ## Other turbulence statistics
            "TaylorScale":[],
            "ReTaylor":[],
            "Re":[],
            "DissRate":[],
            "DissRateNorm":[],
            "EddyTurnOverTime": [],
            "KolTimeScale":[],
            "KolLenScale":[],
            
        }
        self.calculators = []
    
    def register_calculator(self, func):
        self.calculators.append(func)
    
    def step(self, it, Uh, **kwargs):
        self.add_to_history("tt", it * self.dt)
        print('+'*50)
        print(f"****    Simulating at step-{it} | t = {it * self.dt:.6e}.    ****")
        
        for calc in self.calculators:
            calc(self, it, Uh, **kwargs)
    
    def add_to_history(self, key, value):
        if key == "kk" or key == "Deltar":
            self.history[key] = value
        else:
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            
            
###################################################
### Compute energy spectrum
###################################################
def computeEnergyAndSpectrum_FULL(pp, it, Uh_prev, **kwargs):
    Ifadd2his = kwargs.get('Ifadd2his', False)
    Ek, kk = calEnergySpectrum_FULL(Uh_prev, pp.K2, pp.N)
    Hk, _  = calHelicitySpectrum_FULL(Uh_prev, pp.KK, pp.K2, pp.N)
    E, U_rms = calKineticEnergyBySpectrum_FULL(Ek, kk)
    H, _  = calKineticEnergyBySpectrum_FULL(Hk, kk)
    if Ifadd2his:
        pp.add_to_history('kk', kk)
        pp.add_to_history('Ek', Ek)
        pp.add_to_history('Hk', Hk)
        pp.add_to_history('Energy', E)
        pp.add_to_history('Helicity', H)
        pp.add_to_history('U_rms', U_rms)
    print(f"****    Energy = {E[0]:.6e}; Helicity = {H[0]:.6e}.    ****")
    
def calKineticEnergyBySpectrum_FULL(Ek, kk):
    """This function is for computing energy k"""
    # E = 1/2 ∫ u·u dV
    E = np.trapz(Ek, kk[None, ...], axis=1) # [B, ]
    urms = np.sqrt(2. * E / 3.)
    return  E, urms

def calHelicitySpectrum_FULL(Uh, KK, K2, kmax):
    """This function is for computing energy spectrum Ek
    Uh: [B, 3, N, N, N//2+1]
    K2: [N, N, N//2+1]
    """
    B, _, N, _, _ = Uh.shape
    cross_r = torch.cross(KK.unsqueeze(0), Uh.real, dim=-4)
    cross_i = torch.cross(KK.unsqueeze(0), Uh.imag, dim=-4)
    Omegah = -cross_i + 1j * cross_r
    Hr = 0.5 * torch.sum( torch.conj(Uh) * Omegah + torch.conj(Omegah) * Uh, dim=1).real   # [B, N, N, N//2+1]

    weight = torch.ones_like(K2)
    weight[..., 1:N//2] *= 2
    Hr_weighted = Hr * weight   # [B, N, N, Nkz]
    
    Kmag = torch.sqrt( K2 ) # [N, N, N//2+1]
    Kmag_idx = torch.round(Kmag).long().flatten() 

    Hr_flat = Hr_weighted.view(B, -1) # [B, N*N*(N//2+1)]
    Hk = torch.zeros( B, int(kmax) + 1,  device=Uh.device) # [B, kmax+1]
    for b in range(B):
        Hk[b] = torch.bincount(Kmag_idx, weights=Hr_flat[b], minlength=int(kmax)+1)[0:int(kmax)+1]
    
    Hk[:, 0] = 0.0 # mean energy at k = 0 is ZERO
    return  Hk.cpu().numpy(), np.arange(int(kmax)+1)

def calEnergySpectrum_FULL(Uh, K2, kmax, p=2):
    """This function is for computing energy spectrum Ek
    Uh: [B, 3, N, N, N//2+1]
    K2: [N, N, N//2+1]
    """
    B, _, N, _, _ = Uh.shape
    Er = 0.5 * torch.sum( torch.abs(Uh)**p, dim=1 )   # [B, N, N, N//2+1]
 
    weight = torch.ones_like(K2)
    weight[..., 1:N//2] *= 2
    Er_weighted = Er * weight   # [B, N, N, Nkz]
    
    Kmag = torch.sqrt( K2 ) # [N, N, N//2+1]
    Kmag_idx = torch.round(Kmag).long().flatten() 

    Er_flat = Er_weighted.view(B, -1) # [B, N*N*(N//2+1)]
    Ek = torch.zeros( B, int(kmax) + 1,  device=Uh.device) # [B, kmax+1]
    for b in range(B):
        Ek[b] = torch.bincount(Kmag_idx, weights=Er_flat[b], minlength=int(kmax)+1)[0:int(kmax)+1]
    
    Ek[:, 0] = 0.0 # mean energy at k = 0 is ZERO
    return  Ek.cpu().numpy(), np.arange(int(kmax)+1)

def computeForcedAdjustment_FULL(pp, it, Uh_prev, **kwargs):
    """Return the low-wavenumber feedback coefficient for ``f_hat = -c*u_hat``.

    The coefficient balances viscous dissipation and relaxes the total kinetic
    energy to ``ForcedEnergy`` over one time step. All inner products use the
    full-spectrum Parseval weights represented by the stored RFFT half-spectrum.
    """
    Ifadd2his = kwargs.get('Ifadd2his', False)
    if pp.IFforced:
        # Sum over velocity components first, then restore the omitted negative-kz
        # partners of the RFFT representation. The zero and Nyquist planes are
        # self-conjugate and therefore retain unit weight.
        modal_norm = torch.sum(torch.abs(Uh_prev)**2, dim=1) # [B, N, N, N//2+1]
        if pp.N % 2 == 0:
            modal_norm[..., 1:-1] *= 2.0
        else:
            modal_norm[..., 1:] *= 2.0

        sum_dims = (-3, -2, -1)
        E = 0.5 * torch.sum(modal_norm, dim=sum_dims) # [B,]
        DissRate = pp.nu * torch.sum(
            modal_norm * pp.K2.unsqueeze(0), dim=sum_dims
        ) # [B,]

        # The mean mode is excluded because Equation_NS3d sets its residual to zero.
        In_mask = (
            (pp.K2 > 0.0) & (pp.K2 < pp.ForceLowerAtKc**2)
        ).to(dtype=modal_norm.dtype)
        NormIn = torch.sum(
            modal_norm * In_mask.unsqueeze(0), dim=sum_dims
        ) # [B,] = 2 * kinetic energy in the forced band

        if torch.any(NormIn <= 0.0):
            raise RuntimeError("Cannot apply low-wavenumber forcing: forced-band energy is zero.")

        target_energy = torch.as_tensor(
            pp.ForcedEnergy, dtype=E.dtype, device=E.device
        )
        cc = ((E - target_energy) / pp.dt - DissRate) / NormIn # [B,]

        if Ifadd2his:
            pp.add_to_history('cc', cc)
        return cc
    
###################################################
### compute turbulence_stats
###################################################

def computeTurbulenceStats_FULL(pp, it, Uh_prev, **kwargs):
    if pp.nu == 0.0:
        return
    Ek = pp.history['Ek'][-1] # [B,kmax+1]
    kk = pp.history['kk'] # [kmax+1]
    U_rms = pp.history['U_rms'][-1] # [B, ]

    # Calculate dissipation rate first (needed for Taylor scale)
    DissRate = calDissRate_FULL(Ek, kk, pp.nu) # [B, ]
    IntegralLength = calIntergralLength_FULL(Ek, kk, U_rms) # [B, ]
    Taylorscale = (15. * pp.nu * U_rms**2 / DissRate)**0.5
    ReTaylor = U_rms * Taylorscale / pp.nu
    Re = U_rms * IntegralLength / pp.nu
    DissRateNorm = DissRate * IntegralLength / U_rms**3
    EddyTurnOverTime = IntegralLength / U_rms
    KolTimeScale = (pp.nu / DissRate)**0.5
    KolLenScale = (pp.nu**3 / DissRate)**0.25

    pp.add_to_history('Re', Re)
    pp.add_to_history('ReTaylor', ReTaylor)
    
    pp.add_to_history('IntegralLen', IntegralLength)
    pp.add_to_history('TaylorScale', Taylorscale)
    pp.add_to_history('KolLenScale', KolLenScale)
    
    pp.add_to_history('EddyTurnOverTime', EddyTurnOverTime)
    pp.add_to_history('KolTimeScale', KolTimeScale)
    
    pp.add_to_history('DissRate', DissRate)
    pp.add_to_history('DissRateNorm', DissRateNorm)
    
    
    print(f"****    Re = {Re[0]:.6e}.    ****")
    print(f"****    Re_taylor = {ReTaylor[0]:.6e}.    ****")
    print(f"****    kmax * KolmogorovLenScale = {KolLenScale[0] * pp.kmax:.6e}.    ****")
    
    print(f"****    IntegralLength = {IntegralLength[0]:.6e}.    ****")
    print(f"****    TaylorLenScale = {Taylorscale[0]:.6e}.    ****")
    print(f"****    KolmogorovLenScale = {KolLenScale[0]:.6e}.    ****")
    
    print(f"****    Eddy Turnover Time = {EddyTurnOverTime[0]:.6e}.    ****")
    print(f"****    Dissipation rate = {DissRate[0]:.6e}.    ****")
    print(f"****    Normalized Dissipation rate = {DissRateNorm[0]:.6e}.    ****")
    
def calDissRate_FULL(Ek, kk, nu, kmax=None):
    """This function is for computing dissipation rate"""
    if kmax is not None:
        kk_ = kk[kk < kmax][None, ...]
        Ek_ = Ek[...,kk < kmax]
    else:
        kk_ = kk[None, ...]
        Ek_ = Ek
    epsilon = 2 * nu * np.trapz( kk_ * kk_ * Ek_, kk_, axis=1)
    return epsilon


def calIntergralLength_FULL(Ek, kk, Urms):
    """This function is for computing Integral length scale L"""
    Ek_overk = Ek / (kk  + 1e-10) # [B, kmax+1]
    Ek_overk[:, 0] = 0.0
    L  = np.pi  / (2. * Urms**2) * np.trapz( Ek_overk, kk[None, ...], axis=1 )
    return L  # [B, ]
    
    
def computePostPRO_FULL(pp, it, U, **kwargs):
    StructFnc2, Deltar = calStructureFnc_FULL(U, pp.ll, pp.N, 2)
    StructFnc3, _ = calStructureFnc_FULL(U, pp.ll, pp.N, 3)
    
    Uh = rFFT3d_torch_user(U)
    CorrFnc, kk = calCorrelationFnc_FULL(Uh, pp.K2, pp.N)
    dUdx = rIFFT3d_torch_user(1j * pp.KK[0] * Uh[:,:, 0, ...])
    pdf, bins_edges, _ = calPDF_FULL(dUdx.cpu().numpy().reshape(-1,1), bins=100)


    pp.add_to_history('tt', pp.dt * it)
    pp.add_to_history('Deltar', Deltar)
    pp.add_to_history('kk', kk)
    pp.add_to_history('StructFnc2', StructFnc2)
    pp.add_to_history('StructFnc3', StructFnc3)
    pp.add_to_history('CorrFnc', CorrFnc)
    pp.add_to_history('bins', bins_edges)
    pp.add_to_history('pdf', pdf)
    print(f"Finished-{it}!")
    
    
def calSF_FULL(Uh_prev, KK):
    """This function is for computing skewness and kurtosis"""
    Ux = Uh_prev[..., 0, ...] # [B, N, N, N//2+1]
    KX = KK[0, ...] #[N, N, N//2+1]
    
    dUdxh = 1j * KX.unsqueeze(0) * Ux # [B, N, N, N//2+11]
    dUdx = rIFFT3d_torch_user(dUdxh) # [B, N, N, N]
    
    S = torch.mean( dUdx**3, dim=(-1, -2, -3))/ torch.mean( dUdx**2,dim=(-1, -2, -3))**1.5
    F = torch.mean( dUdx**4, dim=(-1, -2, -3))/ torch.mean( dUdx**2,dim=(-1, -2, -3))**2.0
    return S, F #[B, ]

def calStructureFnc_FULL(U, ll, N, order=2, Direction="x"):
    '''
    U: [T, B, 3, N, N, N]
    '''
    
    match Direction:
        case "x": #means x direction
            DD = np.zeros((N//2))
            
            for i in range(1,N // 2):  # x direction priodic
                U_trans_x = TransferMatrix_alongAxis(U[:, :, 0, ...], 2, i)
                CorShot = (U[:,:, 0,...] - U_trans_x)**order
                DD[i] = Cal_Stat(CorShot,[0,1,2,3,4])
        case "t":
            
            pass
        case _:
            print('Now, only can cal D')
            
    return DD, ll[0:N//2].cpu().numpy()

# def calCorrelationFnc_FULL(Uh, K2, kmax):
#     '''
#     Uh shape [T, B, 3, N, N, N//2+1]
#     K2: [1, N, N, N//2+1]
#     '''
    
#     T, B, _, N, _, _ = Uh.shape
    
#     weight = torch.ones_like(K2)
#     weight[..., 1:N//2] *= 2
    
#     Kmag = torch.sqrt( K2 ) # [N, N, N//2+1]
#     Kmag_idx = torch.round(Kmag).long().flatten() 
    
#     RR = torch.zeros( T, int(kmax) + 1,  device=Uh.device) # [B, kmax+1]
#     for it in range(T):
#         if it == 0:
#             RR_ =  torch.mean( Uh[:,:,:,...] * Uh[:,:,:,...].conj(), dim= (0, 1, 2) ) # [N, N, N//2]
#         else:
#             RR_ =  torch.mean( Uh[0:-it,:,:,...] * Uh[it: , :, :,...].conj(), dim= (0, 1, 2) ) # [N, N, N//2]
#         RR_weighted = RR_.real * weight  # [N, N, N//2+1]
#         RR_weighted_flat = RR_weighted.view(-1) # [N*N*(N//2+1)]
#         RR[it] = torch.bincount(Kmag_idx, weights=RR_weighted_flat, minlength=int(kmax)+1)[0:int(kmax)+1]


    return  RR.cpu().numpy(), np.arange(int(kmax)+1)

def calPDF_FULL(data, bins, weights=None):
    std = np.std(data)
    
    hist, bin_edges = np.histogram(data / std, range=(-25, 25), bins=bins, density=True, weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    return hist, bin_centers, std
    
    

    

                


if __name__ == '__main__':
    data = np.random.normal(0, 1, 10000000) * 100  # Example with ten million samples.
    # pdf, bin_centers, var = calcualte_pdf(data, bins=100)  
    
    

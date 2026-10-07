import torch
import numpy as np
###
from tools.FourierTransformation import FFTfreq_torch_user, IFFT3d_torch_user

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class IsotropicTurbulenceGenerator:
    def __init__(self, N=128, k0=8, A=0.00013):
        """
        Initial HIT
        
        Arg:
        N: number of vertex
        k0: energy peak
        A: Constant
        """
        self.N = N
        self.A = A
        self.k0 = k0
        self.pi = np.pi
        
        # compute total energy, and time for eddy turn over
        self.Ek0 = 3.0 * A / 64.0 * np.sqrt(2.0 * self.pi) * (k0 ** 5)
        self.tao = np.sqrt(32.0 / A) * (2.0 * self.pi) ** 0.25 * (k0 ** (-3.5))
        
        print(f"****  Eddy turnover time = {self.tao}  ****")
        print(f"****  Total energy Ek0 = {self.Ek0}  *****")
    
    
    
    def generate_velocity_field(self, device):
        """generate velocity field"""

        dl = 2. * np.pi / self.N
        kx = ky = kz = (2 * np.pi / dl) * FFTfreq_torch_user(self.N).to(device)  #[-N/2, N/2 -1]
        Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        K_squared = Kx**2 + Ky**2 + Kz**2
        K_squared[ K_squared == 0] = 1
        
        U_hat = torch.zeros((self.N, self.N, self.N), dtype=torch.complex64, device=device)
        V_hat = torch.zeros((self.N, self.N, self.N), dtype=torch.complex64, device=device)
        W_hat = torch.zeros((self.N, self.N, self.N), dtype=torch.complex64, device=device)
        
        
        Ak = torch.sqrt(torch.tensor(2.0/3.0)) * torch.sqrt( self.A * K_squared * torch.exp(-2.0 * K_squared / (self.k0**2)) / (4.0 * self.pi) )
                        
        phi1 = 2 * self.pi * torch.rand((self.N, self.N, self.N), device=device)
        phi2 = 2 * self.pi * torch.rand((self.N, self.N, self.N), device=device)
        phi3 = 2 * self.pi * torch.rand((self.N, self.N, self.N), device=device)
                
        v1 = Ak * torch.exp(1j * phi1)
        v2 = Ak * torch.exp(1j * phi2)
        v3 = Ak * torch.exp(1j * phi3)
                        

                        
        # project                         
        U_hat =  v1 -  Kx * ( v1 * Kx + v2 * Ky + v3 * Kz) / K_squared 
        V_hat =  v2 -  Ky * ( v1 * Kx + v2 * Ky + v3 * Kz) / K_squared 
        W_hat =  v3 -  Kz * ( v1 * Kx + v2 * Ky + v3 * Kz) / K_squared 
        
        U_hat[0,0,0] = 0.; V_hat[0,0,0] = 0.; W_hat[0,0,0] = 0.
        
        U = torch.real(IFFT3d_torch_user(U_hat))
        V = torch.real(IFFT3d_torch_user(V_hat))
        W = torch.real(IFFT3d_torch_user(W_hat))
        

        Er = 0.5 * torch.mean(U**2 + V**2 + W**2)
        sr = torch.sqrt(self.Ek0 / Er)
        
        U = U * sr
        V = V * sr
        W = W * sr
        
        UU = torch.stack((U,V,W), dim=0).unsqueeze(0)
        
        return  UU
    
    def visualize_slice(self):
        import matplotlib.pyplot as plt
        
        UU = self.generate_velocity_field()
        
        
        # Select the center slice.
        z_slice = self.N // 2
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(2,3,1)
        im1 = ax1.contourf(UU[0,0, ...].cpu()[:, :, z_slice])
        
        ax2 = fig.add_subplot(2,3,2)
        im2 = ax2.contourf(UU[0,1, ...].cpu()[:, :, z_slice])
        
        ax3 = fig.add_subplot(2,3,3)
        im3 = ax3.contourf(UU[0,2, ...].cpu()[:, :, z_slice])
        
    
        plt.tight_layout()
        plt.show()


# Usage example.
if __name__ == "__main__":
    # Create the generator.
    generator = IsotropicTurbulenceGenerator(N=128, k0=8, A=0.00013)
    
    
    # Visualize the field (optional).
    generator.visualize_slice()

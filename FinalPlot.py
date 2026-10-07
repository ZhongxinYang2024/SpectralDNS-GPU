import matplotlib.pyplot as plt
from tools.FourierTransformation import rIFFT3d_torch_user
import numpy as np


def outputPlot(History, ns3d):
    ## post processing
    fig = plt.figure()
    ax1 = fig.add_subplot(3,4,1)
    ax1.plot(History["tt"], History["Energy"])
    ax1.set_xlim([0, ns3d.Ttotal])
    ax1.set_ylim([0, np.max(History["Energy"])])
    
    if ns3d.nu != 0:
        ax2 = fig.add_subplot(3,4,2)
        ax2.plot(History["tt"], History["ReTaylor"])
    
        ax3 = fig.add_subplot(3,4,3)
        ax3.plot(History["ReTaylor"], History["DissRateNorm"])

    
        ax5 = fig.add_subplot(3,4,5)
        for Ek in History["Ek"]:
            ax5.loglog(kk[1:] * KolLenScale, Ek[1:])
        ax5.loglog(kk[1:] * KolLenScale, 1.62 * DissRate**(2/3) * kk[1:]**(-5/3), linestyle='--', color='gray')
        ax5.set_xlabel("k * eta")
        
        ax6 = fig.add_subplot(3,4,6)
        ax7 = fig.add_subplot(3,4,7)
    
    ## Coutorf
    UU = rIFFT3d_torch_user(Uh_prev)[0]  
    ax9 = fig.add_subplot(3,4,9)
    contour9 = ax9.contourf(UU[0, ...].cpu()[:, :, ns3d.N // 4])
    plt.colorbar(contour9, ax=ax9)  # Add a color bar to ax9.

    ax10 = fig.add_subplot(3,4,10)
    contour10 = ax10.contourf(UU[1, ...].cpu()[:, :, ns3d.N // 4])
    plt.colorbar(contour10, ax=ax10)  # Add a color bar to ax10.

    ax11 = fig.add_subplot(3,4,11)
    contour11 = ax11.contourf(UU[2, ...].cpu()[:, :, ns3d.N // 4])
    plt.colorbar(contour11, ax=ax11)  # Add a color bar to ax11.
    
    
    
    plt.tight_layout()
    plt.show()        

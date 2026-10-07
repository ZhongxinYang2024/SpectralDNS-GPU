import torch
import torch.fft

def rFFTfreq_torch_user(N):
    return torch.fft.rfftfreq(N)

def FFTfreq_torch_user(N):
    return torch.fft.fftfreq(N)

## for 3d data
def FFT3d_torch_user(data_phys):
    return torch.fft.fftn(data_phys, dim=(-3,-2,-1), norm='forward')

def IFFT3d_torch_user(data_freq):
    return torch.fft.ifftn(data_freq, dim=(-3, -2, -1), norm='forward')

def rFFT3d_torch_user(rdata_phys):
    return torch.fft.rfftn(rdata_phys, dim=(-3, -2, -1), norm='forward')

def rIFFT3d_torch_user(rdata_freq):
    return torch.fft.irfftn(rdata_freq, dim=(-3, -2, -1), norm='forward')


## for 2d data
def FFT2d_torch_user(data_phys):
    return torch.fft.fftn(data_phys, dim=(-2,-1), norm='forward')

def IFFT2d_torch_user(data_freq):
    return torch.fft.ifftn(data_freq, dim=( -2, -1), norm='forward')

def rFFT2d_torch_user(rdata_phys):
    return torch.fft.rfftn(rdata_phys, dim=( -2, -1), norm='forward')

def rIFFT2d_torch_user(rdata_freq):
    return torch.fft.irfftn(rdata_freq, dim=( -2, -1), norm='forward')



def compute_3d_fft_ifft():
    # ========== 1. Create example three-dimensional data ==========
    # Use a 4x4x4 complex tensor as the example input.
    # Real-valued input is also valid, although the FFT returns complex values.
    batch_size = 2
    height, width, depth = 4, 4, 4
    
    # Method 1: create random complex-valued data.
    real_data = torch.randn(batch_size, height, width, depth).to(device='cuda')
    imag_data = torch.randn(batch_size, height, width, depth).to(device='cuda')
    complex_data = torch.complex(real_data, imag_data)

    fft_result = FFT3d_torch_user(real_data)
    ifft_result = IFFT3d_torch_user(fft_result)
    diff = torch.abs(real_data - ifft_result.real)
    print(f"\nMaximum reconstruction error: {diff.max().item():.6e}")
    print(f"Mean reconstruction error: {diff.mean().item():.6e}")
    
    complex_fft = FFT3d_torch_user(complex_data)
    complex_ifft = IFFT3d_torch_user(complex_fft)
    complex_diff = torch.abs(complex_data - complex_ifft)
    print(f"Maximum complex-data reconstruction error: {complex_diff.max().item():.6e}")
    


    rfft_result = rFFT3d_torch_user(real_data)
    print(f"rfftn output shape: {rfft_result.shape}")  # The final dimension is halved.
    irfft_result = rIFFT3d_torch_user(rfft_result)
    print(f"irfftn output shape: {irfft_result.shape}")
    
    rfft_diff = torch.abs(real_data - irfft_result)
    print(f"Maximum rfftn/irfftn reconstruction error: {rfft_diff.max().item():.6e}")
    



if __name__ == "__main__":
    # Run the example.
    results = compute_3d_fft_ifft()
    

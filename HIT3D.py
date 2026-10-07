r'''

@Author: Zhongxin Yang (zhongxinyang@stu.pku.edu.cn)

The code is developed for solving 3d incompressible homogeneous isotropic turbulence flow in  (2 \pi)^3 cubic

'''
import torch
import numpy as np
import copy
import os
import pickle

## import user def lib
from tools.FourierTransformation import rFFT3d_torch_user, rIFFT3d_torch_user
from tools.FourierTransformation import FFTfreq_torch_user, rFFTfreq_torch_user
from src.BC import BoundaryCondition
from src.InitialC import IsotropicTurbulenceGenerator
from PostProcess_Full import computeEnergyAndSpectrum_FULL
from PostProcess_Full import computeForcedAdjustment_FULL
from PostProcess_Full import computeTurbulenceStats_FULL
from PostProcess_Full import PostProcessor_FULL
from PostProcess_Full import computePostPRO_FULL
from tools.utils import make_dir, save_to_tecplot_fast
from src.TimeScheme import RK2, RK4
from NSeq import Equation_NS3d, apply_projection

class HITsolver3D_effective:
    def __init__(self, config, Rule='sim', U_prev = None, last_time=0, num_stat=100, batch_size_stat=1):
        """
        kx: [-N/2 , N/2 - 1]; ky:[-N/2, N/2-1]; kz:[0, N/2]
        Args:
            N (int, optional):  the number in each direction, prefer 2**n. Defaults to 128.
            L (float, optional): the length of the domain in each direction. Defaults to 2.0*np.pi.
            nu (float, optional): viscosity coefficent. Defaults to 0.001.
            dt (float, optional): time step. Defaults to 0.01.
            Ttotal (float, optional): total simulation time.
            Tscheme (str, optional): time intergration scheme
            dealias_method (str, optional): method to dealias
            device (str, optional): device to run the simulation on. Defaults to 'cuda'.
            
        """
        B = config.get('B', 1)
        self.N = config['N']
        self.L = config['L']
        self.nu = config['nu']                 
        self.last_time = last_time
        self.TschemeName = config['TschemeName']
        self.dt = config['dt']
        self.Ttotal = config['Ttotal']
        self.Nt = int(config['Ttotal'] / config['dt']) + 1
        self.Step_save = config['Step_save']
        self.Step_save_begin = config.get('Step_save_begin',0)
        self.Step_output = config['Step_output']
                
        self.IFforced = config['IFforced']
        self.ForcedEnergy = config['ForcedEnergy']
        self.ForceLowerAtKc = config['ForceLowerAtKc']
    
        self.dealias_method = config['dealias_method']
        
        
        device_from_config = config['device']
        self.device = device_from_config if torch.cuda.is_available() and 'cuda' in device_from_config else 'cpu'
        self.savefolder = config['savefolder']
        self.savefolder_std = config.get('savefolder_std', self.savefolder)
        self.stat_id_begin = config.get('stat_id_begin', 0)
        make_dir(self.savefolder)
        
        ###################################
        ##### Determin time scheme
        ###################################
        if self.TschemeName == 'RK4':
            self.Tscheme = RK4(self.dt)
        elif self.TschemeName == 'RK2':
            self.Tscheme = RK2(self.dt)
        
        
        self.bc = BoundaryCondition('None')
        self.eq = Equation_NS3d()
        
        print('****    I. Initial grid    ****')
        self._setup_grid() ## initial grid in physical space and wavenamber space
        
        
        print('****    II. Initial Projector operator    ****')
        self._setup_projector()  ## initial projector
        
        
        if Rule == 'sim':
            print('****    III. Initial flow field    ****')
            if U_prev is None:
                generator = IsotropicTurbulenceGenerator(N=self.N, k0=8, A=0.00013)
                U_prev = generator.generate_velocity_field(self.device)
                U_prev = U_prev.repeat(B, 1, 1, 1, 1) 
            else: 
                U_prev = U_prev.to(self.device)
            self.Uh_prev = rFFT3d_torch_user(U_prev); del U_prev  # h denote hat, means in wavenumber
            self.Uh_prev[..., 0,0,0] = 0.0 + 0.0j
            self.Uh_prev =   apply_projection(self.Uh_prev.real, self.KK, self.inv_K2_safe) + \
                                1j * apply_projection(self.Uh_prev.imag, self.KK, self.inv_K2_safe) 
            print('****    IV. Running Simulations   ****')
            self._Run_Simulation()
        
        
        elif Rule == "post":
            print('****    V. Post-processing   ****')
            self._Post_Processing(num_stat, batch_size_stat)
        else:
            raise ValueError("Rule must be either 'sim' or 'post'.")
        
        
    def _setup_grid(self):
        assert self.N % 2 == 0, "-- Error --.  N must be deivisble by 2"
        ## physical grid
        self.ll  = torch.linspace(0, self.L, self.N+1)[0:self.N]
        self.dl = self.L / (self.N)
        # self.X, self.Y, self.Z = torch.meshgrid(ll, ll, ll, indexing='ij')
        
        ## wavenumber grid
        self.kx = self.ky = (2 * np.pi / self.dl) * FFTfreq_torch_user(self.N).to(self.device)  #[-N/2, N/2 -1]
        self.kz_pos = (2 * np.pi / self.dl) * rFFTfreq_torch_user(self.N).to(self.device) #[0, N/2]
        KK_grid= torch.meshgrid(self.kx, self.ky, self.kz_pos, indexing='ij')
        self.KK =  torch.stack(KK_grid, dim=0); del KK_grid
        self.K2  = self.KK[0]**2 + self.KK[1]**2 + self.KK[2]**2 # [N, N, N//2+1]
        self.K2_safe = copy.deepcopy(self.K2)
        self.K2_safe[self.K2_safe ==0] = 1 # to aviod divide by zero
        
        if self.dealias_method == '3_2':
            self.Np = int(self.N // 2) * 3; self.dlp = self.L / (self.Np)
            self.pad_start = self.N // 2; self.pad_end = - self.N // 2 + 1;
            
            self.kxp = self.kyp = (2 * np.pi / self.dlp) * FFTfreq_torch_user(self.Np).to(self.device) 
            self.kzp_pos = (2 * np.pi / self.dlp) * rFFTfreq_torch_user(self.Np).to(self.device) 
            KKp_grid = torch.meshgrid(self.kxp, self.kyp, self.kzp_pos, indexing='ij')
            self.KKp =  torch.stack(KKp_grid, dim=0); del KKp_grid
            self.dealias_var = {'Np': self.Np, 'KKp': self.KKp,
                                'pad_start': self.pad_start, 'pad_end': self.pad_end}
            self.kmax = self.N // 2
        elif self.dealias_method == 'phase_shift':
            # ---------------------------------------------------------
            # Phase-Shift dealias method (Patterson-Orszag method)
            # ---------------------------------------------------------
            # 1. Truncation Mask:  kmax^2 = 2/9 * N^2
            kmax_sq_phase = 2.0 * (self.N / 3.0)**2
            self.kmax = np.sqrt(kmax_sq_phase)  # = sqrt(2)/3 * N ≈ 0.4714N
            self.mask_phase = (self.K2 <= kmax_sq_phase).float()

            # 2. Phase-shift: shift half grid (dx/2 = (L / N) / 2)
            # displacement = kx * (L/2N) + ky * (L/2N) + kz * (L/2N)
            shift_phase = (self.KK[0] + self.KK[1] + self.KK[2]) * (self.dl / 2.) # [N, N, N//2]
        
            self.shift_factor = torch.exp(1j * shift_phase)
            self.unshift_factor = torch.exp(-1j * shift_phase)

            self.dealias_var = {
                'mask_phase': self.mask_phase,
                'shift_factor': self.shift_factor,
                'unshift_factor': self.unshift_factor
            }
        elif self.dealias_method == '2_3':
            self.kmax = self.N / 3.
            self.mask23 = (self.K2 <= self.kmax**2).float()
            self.dealias_var = {'mask23': self.mask23}
            

    def _setup_projector(self):
        # P_ij = delta_ij - k_i k_j / k^2
        self.inv_K2_safe = 1. / self.K2_safe
        del self.K2_safe
            
            
    def _Run_Simulation(self):
        ## load current_state       
        Uh_prev = self.Uh_prev; del self.Uh_prev
        
        kwargs = {"dealias_method": self.dealias_method, "dealias_var": self.dealias_var,
                  "nu": self.nu, 
                  "K2": self.K2, "KK": self.KK,
                  "inv_K2_safe": self.inv_K2_safe,
                  "IFforced": self.IFforced,  "ForcedConst":0.0,  "ForceLowerAtKc": self.ForceLowerAtKc}
        
        
        pp = PostProcessor_FULL(dt=self.dt, N=self.N, K2=self.K2, 
                                KK=self.KK, nu=self.nu, kmax=self.kmax,
                                ll=self.ll, IFforced=self.IFforced, 
                                ForcedEnergy=self.ForcedEnergy, ForceLowerAtKc=self.ForceLowerAtKc)
        
        pp.register_calculator(computeEnergyAndSpectrum_FULL)
        pp.register_calculator(computeTurbulenceStats_FULL)
        
        with torch.no_grad(): 
            for it in range(self.Nt):    
                ### =====================================================
                ###  1. compute force constant
                ### =====================================================
                if self.IFforced:
                    kwargs['ForcedConst'] = computeForcedAdjustment_FULL(pp, it, Uh_prev)
                
                ### =====================================================
                ### 2. output in screen
                ### =====================================================
                if it % self.Step_output == 0:
                    pp.step(it, Uh_prev, **{'Ifadd2his':True})
                
                
                ### =====================================================
                ### 3. step forward
                ### =====================================================
                Uh_curr = self.Tscheme.step(Uh_prev, self.eq, self.bc, **kwargs)
                
                
                ### =====================================================
                ### 4. saveing in physical space
                ### =====================================================
                if (it % self.Step_save  == 0):
                    with open(os.path.join(self.savefolder, f'History_{self.nu*1000:.3f}m.json'), 'wb') as f:
                        pickle.dump(pp.history, f)
                    
                    if it > self.Step_save_begin:
                        UU = rIFFT3d_torch_user(Uh_prev).detach().clone().cpu().numpy()
                        np.savez_compressed(os.path.join(self.savefolder, f"YZXDNS-{str(it+self.last_time).zfill(8)}.npz"),
                                            tt = self.dt * it,
                                            nu = self.nu,
                                            xx = self.ll.cpu().numpy(),
                                            UU = UU)
                        gap = self.N // self.N
                        save_to_tecplot_fast(os.path.join(self.savefolder, "current_flow_field.dat"),
                                        ll=self.ll[::gap].cpu().numpy(),
                                        variables=[UU[0,0,::gap, ::gap, ::gap], UU[0,1,::gap, ::gap, ::gap], UU[0,2,::gap, ::gap, ::gap]],
                                        var_names=['U', 'V', 'W'])
                

                ############################################
                ### 5. double check here
                ###########################################
                Uh_prev = Uh_curr
            

            
                                
                
        

    def _Post_Processing(self, num_stat, batch_size_stat):
        pp = PostProcessor_FULL(self.dt, self.N, self.K2, self.KK, self.nu, self.kmax, self.ll,
                           IFforced=self.IFforced, ForcedEnergy=self.ForcedEnergy, ForceLowerAtKc=self.ForceLowerAtKc)
        
        flows_file = []
        for it in range(0, self.Nt, self.Step_save):
            if it < self.stat_id_begin:
                continue
            try:
                filename = os.path.join( self.savefolder_std, f"YZXDNS-{str(it).zfill(8)}.npz")
                if not os.path.exists( filename  ): continue
                flows_file.append( filename  )
                print(f"Add {str(it).zfill(8)}")
                if len(flows_file) > num_stat:
                    break
            except:
                print(f"{it} not found!")


        def calCorrelationFnc_ForBigFile(flows_file):
            Kmag = torch.sqrt(pp.K2)
            Kmag_idx = torch.round(Kmag).long().flatten()
            
            weight = torch.full_like(Kmag, 2)
            weight[..., pp.K2 < 0.5] = 1.0
            
            stride = max(1, len(flows_file) // (batch_size_stat * 2))
            origin_indices =[i * stride for i in range(batch_size_stat)]
            
            origin_indices =[i for i in origin_indices if i < len(flows_file)]
            actual_batch_size = len(origin_indices)
            
            # 1. Load each origin file once and keep it in memory without a
            # large torch.cat operation that could exhaust GPU memory.
            flows0_list = []
            for idx in origin_indices:
                currfield = np.load(flows_file[idx])
                # Keep a list of tensors with shape [1, B, 3, N, N, N//2+1].
                U = torch.from_numpy( currfield['UU'] ).to(self.device)
                flows0_list.append( rFFT3d_torch_user(U).unsqueeze(0))

            max_tau_steps = len(flows_file) - max(origin_indices)
            RR = torch.zeros((max_tau_steps, pp.N+1), device=self.device)
            
            print(f"Start processing... Total origins: {actual_batch_size}, Max Tau Steps: {max_tau_steps}")

            # 2. Read the remaining files in one pass to minimize I/O.
            for file_idx in range(len(flows_file)):
                
                # Files earlier than the first origin do not contribute to the delayed calculation.
                if file_idx < min(origin_indices):
                    continue
                    
                # Load each file only once.
                delayed_field = np.load(flows_file[file_idx])
                current_flow = torch.from_numpy(delayed_field['UU']).to(self.device).unsqueeze(0)
                current_flow = rFFT3d_torch_user(current_flow)
                # Determine the lag tau for this file relative to each origin.
                for i, origin_idx in enumerate(origin_indices):
                    if file_idx >= origin_idx:
                        tau = abs(file_idx - origin_idx)
                        
                        if tau < max_tau_steps:
                            RR_pair = torch.mean(flows0_list[i] * current_flow.conj(), dim=(0,1,2)).real # [N, N, N//2+1]
                            
                            # Dividing by actual_batch_size is equivalent to an outer mean(dim=0).
                            RR_pair_weighted_flat = ((RR_pair / actual_batch_size) * weight).flatten()
                            RR[tau, :] += torch.bincount(Kmag_idx, weights=RR_pair_weighted_flat, minlength=pp.N+1)[0:pp.N+1]
                
                # Report progress.
                if file_idx % 10 == 0:
                    print(f"Processed file {file_idx}/{len(flows_file)}")
                    
            return RR.cpu().numpy(), np.arange(pp.N+1)
        
        def calIntermitency_ForBigFile(flows_file):
            from PostProcess_Full import calPDF_FULL
            
            stride = max(1, len(flows_file) // (batch_size_stat * 2))
            origin_indices =[i * stride for i in range(batch_size_stat)]
            
            Kmag = torch.sqrt(pp.K2)
            Kmag_idx = torch.round(Kmag).long()
            
            weight = torch.full_like(Kmag, 2)
            weight[..., pp.K2 < 0.5] = 1.0
            weight_np = weight.unsqueeze(0).repeat(3,1,1,1).cpu().numpy()
            
            bins = 1000
            pdf = np.zeros( (int(pp.kmax) + 1, bins))
            bin_centers = None
            for count, file_idx in enumerate(origin_indices):    
                field = np.load(flows_file[file_idx])
                current_flow = torch.from_numpy(field['UU']).to(self.device) # [1, 3, N, N, N//2+1]
                Uh = rFFT3d_torch_user(current_flow).cpu().numpy()
                
                for ik in range(int(pp.kmax)+1):
                    mask = (Kmag_idx >= ik) & (Kmag_idx < ik+1)
                    if mask.sum() == 0:
                        continue
                    mask_np = mask.cpu().numpy()
                    data = Uh[..., mask_np].real.flatten() # [L]
                    w = weight_np[..., mask_np].flatten()
                    
                    hist, cents, _ = calPDF_FULL(data, bins, weights=w)
                    
                    if bin_centers is None:
                        bin_centers = cents                # Store the bin centers once.
            
                    pdf[ik] = (pdf[ik] * count + hist) / (count + 1)

                if count % 10 == 0:
                    print(f"Processed file {count+1}/{len(origin_indices)}")
                
            return pdf, bin_centers, np.arange(int(pp.kmax)+1)
        
        
        def calStructeInFourier_ForBigFile(flows_file, plst=[2]):
            from PostProcess_Full import calEnergySpectrum_FULL
            
            stride = max(1, len(flows_file) // (batch_size_stat * 2))
            origin_indices =[i * stride for i in range(batch_size_stat)]
        
            Sp = np.zeros( (len(plst), int(pp.kmax) + 1 ))
            
            for count, file_idx in enumerate(origin_indices):    
                field = np.load(flows_file[file_idx])
                current_flow = torch.from_numpy(field['UU']).to(self.device) # [1, 3, N, N, N//2+1]
                Uh = rFFT3d_torch_user(current_flow)
                
                for ip, p in enumerate(plst):
                    Sp_tp, kk = calEnergySpectrum_FULL(Uh, pp.K2, pp.kmax, p=p)
                    Sp[ip] = (Sp[ip] * count + Sp_tp) / (count + 1)

                if count % 10 == 0:
                    print(f"Processed file {count+1}/{len(origin_indices)}")
                
            return Sp, kk
        
        plst = [2,3,4,5,6,8]
        Sp, kk_sp = calStructeInFourier_ForBigFile(flows_file, plst=plst)
        for ip, p in enumerate(plst):
            pp.add_to_history(f'Sp{int(p)}', Sp[ip])
        pp.add_to_history('kk_sp', kk_sp)        
        
        # pdf, bin_centers, kk_pdf = calIntermitency_ForBigFile(flows_file)
        # pp.add_to_history('kk_pdf', kk_pdf)
        # pp.add_to_history('bin_centers', bin_centers)   
        # pp.add_to_history('pdf', pdf)  
        
        CorrFnc, kk = calCorrelationFnc_ForBigFile(flows_file)
        pp.add_to_history('kk', kk)
        pp.add_to_history('CorrFnc', CorrFnc)
        pp.add_to_history('Dtau_Cor', np.linspace(0, CorrFnc.shape[0] * self.dt * self.Step_save, CorrFnc.shape[0], endpoint=False))
        
   
        


        
        with open(os.path.join(self.savefolder_std, f'Statistic_{self.nu*1000:.3f}m.json'), 'wb') as f:
            pickle.dump(pp.history, f)



                 

        

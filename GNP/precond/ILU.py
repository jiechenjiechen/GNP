import torch
import pickle as pkl
from scipy import sparse

from GNP.utils import spsolve_lu


#-----------------------------------------------------------------------------
# ILU
class ILU():

    # A must be sparse csc
    def __init__(self, A, ilu_factors_file, save_ilu_factors, fill_factor=None):
        
        self.device = A.device

        if A.layout != torch.sparse_csc:
            raise Exception('To use ILU, A must be sparse csc')
        
        # A in device, invA_approx in cpu
        #
        # ilu_factors_file = file name with path, save_ilu_factors = False:
        #     Load ILU factors from file
        #
        # ilu_factors_file = file name with path, save_ilu_factors = True:
        #     Compute ILU factors and save
        #
        # ilu_factors_file = None, save_ilu_factors = False:
        #     Compute ILU factors but do not save
        #
        # ilu_factors_file = None, save_ilu_factors = True:
        #     Error

        if ilu_factors_file is not None and save_ilu_factors is False:

            with open(ilu_factors_file, 'rb') as f:
                C = pkl.load(f)
                self.L = C['L']
                self.U = C['U']
                self.perm_c = C['perm_c']
                self.perm_r = C['perm_r']
                
        elif (ilu_factors_file is not None and save_ilu_factors is True) or \
             (ilu_factors_file is None and save_ilu_factors is False):
            
            A = A.to('cpu')
            n = A.shape[0]
            spA = sparse.csc_array((A.values().numpy(),
                                    A.row_indices().numpy(),
                                    A.ccol_indices().numpy()), shape=(n,n))
            if fill_factor is None:
                B = sparse.linalg.spilu(spA)
            else:
                B = sparse.linalg.spilu(spA, fill_factor=fill_factor)
            self.L = B.L.tocsr()
            self.U = B.U.tocsr()
            self.perm_c = B.perm_c
            self.perm_r = B.perm_r

            if save_ilu_factors is True:
                C = {'L':B.L.tocsr(), 'U':B.U.tocsr(),
                     'perm_c':B.perm_c, 'perm_r':B.perm_r}
                with open(ilu_factors_file, 'wb') as f:
                    pkl.dump(C, f)
            
        else:

            raise Exception('Incorrect combination of '
                            'ilu_factors_file and save_ilu_factors')
    
    def apply(self, r): # r is in device
        r = r.to('cpu').numpy()
        z = spsolve_lu(self.L, self.U, r, self.perm_c, self.perm_r)
        z = torch.from_numpy(z).to(self.device)
        return z

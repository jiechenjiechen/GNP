import torch
import pickle as pkl
from scipy import sparse

from GNP.utils import spsolve_lu, extract_block_diagonal


#-----------------------------------------------------------------------------
# Block Jacobi
class BlockJacobi():

    # A must be sparse csc
    def __init__(self, A, bjac_lu_factors_file, save_bjac_lu_factors,
                 block_size=32):

        self.device = A.device
        
        if A.layout != torch.sparse_csc:
            raise Exception('To use BlockJacobi, A must be sparse csc')
        
        # A in device, block Jacobi LU factors in cpu
        #
        # bjac_lu_factors_file = full path, save_bjac_lu_factors = False:
        #     Load block Jacobi LU factors from file
        #
        # bjac_lu_factors_file = full path, save_bjac_lu_factors = True:
        #     Compute block Jacobi LU factors and save
        #
        # bjac_lu_factors_file = None, save_bjac_lu_factors = False:
        #     Compute block Jacobi LU factors but do not save
        #
        # bjac_lu_factors_file = None, save_bjac_lu_factors = True:
        #     Error

        if bjac_lu_factors_file is not None and save_bjac_lu_factors is False:

            with open(bjac_lu_factors_file, 'rb') as f:
                C = pkl.load(f)
                self.L = C['L']
                self.U = C['U']
                self.perm_c = C['perm_c']
                self.perm_r = C['perm_r']
                
        elif (bjac_lu_factors_file is not None and \
              save_bjac_lu_factors is True) or \
              (bjac_lu_factors_file is None and \
               save_bjac_lu_factors is False):

            D = extract_block_diagonal(A, block_size)
            B = sparse.linalg.splu(D)
            self.L = B.L.tocsr()
            self.U = B.U.tocsr()
            self.perm_c = B.perm_c
            self.perm_r = B.perm_r

            if save_bjac_lu_factors is True:
                C = {'L':B.L.tocsr(), 'U':B.U.tocsr(),
                     'perm_c':B.perm_c, 'perm_r':B.perm_r}
                with open(bjac_lu_factors_file, 'wb') as f:
                    pkl.dump(C, f)
            
        else:

            raise Exception('Incorrect combination of '
                            'bjac_lu_factors_file and save_bjac_lu_factors')
    
    def apply(self, r): # r is in device
        r = r.to('cpu').numpy()
        z = spsolve_lu(self.L, self.U, r, self.perm_c, self.perm_r)
        z = torch.from_numpy(z).to(self.device)
        return z

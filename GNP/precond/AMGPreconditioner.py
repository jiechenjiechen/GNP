import torch
from scipy import sparse
from pyamg.blackbox import solver_configuration, solver


#-----------------------------------------------------------------------------
# AMG
class AMGPreconditioner():

    # A must be sparse csc
    def __init__(self, A):
        
        if A.layout != torch.sparse_csc:
            raise Exception('To use AMGPreconditioner, A must be sparse csc')
        
        self.device = A.device
        A = A.to('cpu')
        n = A.shape[0]
        spA = sparse.csc_array((A.values().numpy(),
                                A.row_indices().numpy(),
                                A.ccol_indices().numpy()), shape=(n,n))
        spA = sparse.csr_matrix(spA)
        config = solver_configuration(spA, verb=False)
        mg = solver(spA, config)
        self.M = mg.aspreconditioner()

    def apply(self, r): # r is in device
        r = r.to('cpu').numpy()
        z = self.M * r
        z = torch.from_numpy(z).to(self.device)
        return z

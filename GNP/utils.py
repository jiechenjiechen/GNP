import os
import mat73
import torch
import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from ssgetpy import fetch


#-----------------------------------------------------------------------------
# Load problem of SuiteSparse.
# problem must be in the form group/name.
# Return torch.sparse_csc_tensor in torch.float64 precision in device.
# For the python interface of SuiteSparse, see https://github.com/drdarshan/ssgetpy
def load_suitesparse(location, problem, device):
    
    matrix = fetch(problem, format='MAT', dry_run=True)
    
    if len(matrix) != 0:
        
        location = os.path.abspath(os.path.expanduser(location))
        group, name = problem.split('/')
        fetch(problem, format='MAT', location=os.path.join(location, group))[0]
        try:
            P = sio.loadmat(os.path.join(location, problem))
            A = P['Problem']['A'][0][0]
        except NotImplementedError:
            P = mat73.loadmat(os.path.join(location, problem + '.mat'))
            A = P['Problem']['A']
        del P
        A = torch.sparse_csc_tensor(A.indptr, A.indices, A.data, A.shape,
                                    dtype=torch.float64).to(device)
        return A
    
    else:
        
        raise Exception(f'Unsupported problem {problem}!')

    
#-----------------------------------------------------------------------------
# Scale A by an estimated spectral radius according to the Gershgorin
# circle theorem.
def scale_A_by_spectral_radius(A):
 
    if A.layout == torch.sparse_csc:
        
        absA = torch.absolute(A)
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=A.dtype, device=A.device)
        col_sum = torch.ones(1, m, dtype=A.dtype, device=A.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = A * (1. / gamma.item())
        
    elif A.layout == torch.strided:

        absA = torch.absolute(A)
        row_sum = torch.sum(absA, dim=1)
        col_sum = torch.sum(absA, dim=0)
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = A / gamma
        
    else:

        raise NotImplementedError(
            'A must be either torch.sparse_csc_tensor or torch.tensor')
    
    return outA


#-----------------------------------------------------------------------------
# Extract the diagonal of A.
def extract_diagonal(A):

    if A.layout == torch.sparse_csc:

        n = A.shape[0]
        D = torch.zeros(n, device=A.device, dtype=A.dtype)
        A = A.to_sparse_coo().coalesce()

        indices = A.indices()
        mask = indices[0] == indices[1]
        diagonal_values = A.values()[mask]
        diagonal_indices = indices[0][mask]

        D = D.scatter_add(dim=0, index=diagonal_indices, src=diagonal_values)
    
    elif A.layout == torch.strided:
        
        D = torch.diagonal(A)
        
    else:
        
        raise NotImplementedError(
            'A must be either torch.sparse_csc_tensor or torch.tensor')
    
    return D


#-----------------------------------------------------------------------------
# Extract the block diagonal of A.
# Assume A is torch.sparse_csc, on device.
# The returned D is scipy.sparse.csc_array, on cpu.
def extract_block_diagonal(A, block_size):

    if A.layout != torch.sparse_csc:
        raise Exception('To use BlockJacobi, A must be sparse csc')

    n = A.shape[0]
    A = A.to_sparse_coo().coalesce().to('cpu')

    indices = A.indices()
    mask = (indices[0] // block_size) == (indices[1] // block_size)
    D_values = A.values()[mask]
    D_indices = indices[:, mask]
    
    D = sparse.coo_array((D_values.numpy(),
                          (D_indices[0].numpy(),
                           D_indices[1].numpy())), shape=(n,n))
    
    D = D.tocsc()
    
    return D


#-----------------------------------------------------------------------------
# Replacement of scipy.sparse.linalg.SuperLU.solve().
# Adapted from https://stackoverflow.com/questions/29620809/pickling-scipys-superlu-class-for-incomplete-lu-factorization
def spsolve_lu(L, U, b, perm_c=None, perm_r=None):
    """ an attempt to use SuperLU data to efficiently solve
    Ax = Pr.T L U Pc.T x = b
     - note that L from SuperLU is in CSC format solving for c
       results in an efficiency warning
    Pr . A . Pc = L . U
    Lc = b      - forward solve for c
     c = Ux     - then back solve for x
    """
    if perm_r is not None:
        bb = b.copy()
        bb[perm_r] = b
    c = spsolve_triangular(L, bb, lower=True, unit_diagonal=True)
    x = spsolve_triangular(U, c, lower=False)
    if perm_c is None:
        return x
    else:
        return x[perm_c]


#-----------------------------------------------------------------------------
if __name__ == '__main__':

    # Test spsolve_lu()
    n = 6
    density = 0.25
    A = sparse.random(n, n, density=density)
    A.setdiag(1)
    A = A.tocsc()
    x = np.random.random(n)
    b = A @ x
    
    B = sparse.linalg.spilu(A)
    x1 = B.solve(b)
    x2 = spsolve_lu(B.L, B.U, b, B.perm_c, B.perm_r)
    x3 = spsolve_lu(B.L.tocsr(), B.U.tocsr(), b, B.perm_c, B.perm_r)

    print(A.todense())
    print(B.L.todense())
    print(B.U.todense())
    print(x)
    print(x1)
    print(x2)
    print(x3)

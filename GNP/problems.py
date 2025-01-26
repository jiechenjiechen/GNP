import torch
from scipy import sparse

# Convention: Matrices and right-hand sides are float64.
#
# Convention: Matrices are by default sparse. Functions that generate
# full matrices have suffix _full in their names.
#
# Convention: All functions start with 'gen_'.
#
# Convention: All return data are torch tensors and are on cpu.


#-----------------------------------------------------------------------------
# Matrices
def gen_1d_laplacian(n):
    A = sparse.diags([-1,2,-1], [-1,0,1], shape=(n,n))
    A = A.tocsc()
    A = torch.sparse_csc_tensor(A.indptr, A.indices, A.data,
                                dtype=torch.float64)
    return A

def gen_1d_laplacian_full(n):
    A = sparse.diags([-1,2,-1], [-1,0,1], shape=(n,n))
    A = A.toarray()
    A = torch.tensor(A, dtype=torch.float64)
    return A

def gen_1d_signless_laplacian_full(n):
    A = sparse.diags([+1,2,+1], [-1,0,1], shape=(n,n))
    A = A.toarray()
    A = torch.tensor(A, dtype=torch.float64)
    return A


#-----------------------------------------------------------------------------
# Right-hand sides/solutions
def gen_b_all_ones(n):
    return torch.ones(n, dtype=torch.float64)

def gen_x_all_ones(n):
    return torch.ones(n, dtype=torch.float64)

def gen_b_randn(n):
    return torch.normal(0, 1, size=(n,), dtype=torch.float64)

def gen_x_randn(n):
    return torch.normal(0, 1, size=(n,), dtype=torch.float64)

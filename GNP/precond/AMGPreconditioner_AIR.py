import torch
from scipy import sparse
from pyamg.blackbox import solver_configuration, make_csr
from pyamg.classical import air_solver
from pyamg.aggregation import smoothed_aggregation_solver


#-----------------------------------------------------------------------------
# The following code is adapted from pyamg.blackbox.solver
def solver(A, config):
    """Generate an SA or AIR solver given matrix A and a configuration.

    Parameters
    ----------
    A : array, matrix, csr_matrix, bsr_matrix
        Matrix to invert, CSR or BSR format preferred for efficiency
    config : dict
        A dictionary of solver configuration parameters that is used to
        generate a smoothed aggregation solver or an AIR solver

    Returns
    -------
    ml : smoothed_aggregation_solver or air_solver

    Notes
    -----
    config must contain the following parameter entries for
    smoothed_aggregation_solver: symmetry, smooth, presmoother, postsmoother,
    B, strength, max_levels, max_coarse, coarse_solver, aggregate, keep

    air_solver uses the default parameters

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import solver_configuration,solver
    >>> A = poisson((40,40),format='csr')
    >>> config = solver_configuration(A,verb=False)
    >>> ml = solver_air(A,config)

    """
    # Convert A to acceptable format
    A = make_csr(A)

    # Generate smoothed aggregation solver
    if config['symmetry'] == 'hermitian':
        try:
            return \
                smoothed_aggregation_solver(A,
                                            B=config['B'],
                                            BH=config['BH'],
                                            smooth=config['smooth'],
                                            strength=config['strength'],
                                            max_levels=config['max_levels'],
                                            max_coarse=config['max_coarse'],
                                            coarse_solver=config['coarse_solver'],
                                            symmetry=config['symmetry'],
                                            aggregate=config['aggregate'],
                                            presmoother=config['presmoother'],
                                            postsmoother=config['postsmoother'],
                                            keep=config['keep'])
        except Exception as e:
            raise TypeError('Failed generating smoothed_aggregation_solver') from e
    else:
        try:
            # # Default parameters for air_solver do not work. They
            # # cause "Warning : zero diagonal encountered in Jacobi;
            # # ignored"
            # return \
            #     air_solver(A)
            return \
                air_solver(A,
                           presmoother=config['presmoother'],
                           postsmoother=config['postsmoother'])
        except Exception as e:
            raise TypeError('Failed generating air_solver') from e


#-----------------------------------------------------------------------------
# AMG. This preconditioner uses a variant of the default PyAMG, which
# calls pyamg.aggregation.smoothed_aggregation_solver. We call
# pyamg.classical.air_solver instead when the matrix is unsymmetric.
class AMGPreconditioner_AIR():

    # A must be sparse csc
    def __init__(self, A):
        
        if A.layout != torch.sparse_csc:
            raise Exception('To use AMGPreconditioner_AIR, A must be sparse csc')
        
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

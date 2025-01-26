from GNP.solver import GMRES


#-----------------------------------------------------------------------------
# GMRES as preconditioner (used in FGMRES)
class GMRESPreconditioner():

    # A is torch tensor, either sparse or full
    def __init__(self, A, inner_iters=10, inner_rtol=1e-6):
        self.A = A
        self.inner_iters = inner_iters
        self.inner_rtol = inner_rtol
        self.gmres = GMRES()
    
    def apply(self, r): # r is in device
        z, _, _, _, _ = self.gmres.solve(
            self.A, r, M=None, restart=self.inner_iters,
            max_iters=self.inner_iters, timeout=None,
            rtol=self.inner_rtol, progress_bar=False)
        return z

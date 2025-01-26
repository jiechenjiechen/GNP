from GNP.utils import extract_diagonal


#-----------------------------------------------------------------------------
# Jacobi
class Jacobi():

    def __init__(self, A): # A is torch tensor, either sparse or full
        self.D = extract_diagonal(A)
        self.D[self.D == 0] = 1

    def apply(self, r): # r: float64, in device
        z = r / self.D
        return z

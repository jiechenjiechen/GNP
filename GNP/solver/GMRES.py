import time
import torch
import numpy as np
from tqdm import tqdm
from warnings import warn

# Convention: M \approx inv(A)


#-----------------------------------------------------------------------------
# (Right-)Preconditioned generalized minimal residual method. Can use
# flexible preconditioner. Current implementation supports only a
# single right-hand side.
#
# If timeout is not None, max_iters is disabled.
class GMRES():

    def solve(self, A, b, M=None, x0=None, restart=10, max_iters=100,
              timeout=None, rtol=1e-8, progress_bar=True):

        if progress_bar:
            if timeout is None:
                pbar = tqdm(total=max_iters, desc='Solve')
                pbar.update()
            else:
                pbar = tqdm(desc='Solve')
                pbar.update()
                        
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0
        norm_b = torch.linalg.norm(b)
        hist_abs_res = []
        hist_rel_res = []
        hist_time = []

        n = len(b)
        V = torch.zeros(n, restart+1, dtype=b.dtype).to(b.device)
        Z = torch.zeros(n, restart, dtype=b.dtype).to(b.device)
        H = torch.zeros(restart+1, restart, dtype=b.dtype).to(b.device)
        g = torch.zeros(restart+1, dtype=b.dtype).to(b.device)
        c = torch.zeros(restart, dtype=b.dtype).to(b.device)
        s = torch.zeros(restart, dtype=b.dtype).to(b.device)
        
        tic = time.time()

        # Initial step
        r = b - A @ x
        beta = torch.linalg.norm(r)
        abs_res = beta
        rel_res = abs_res / norm_b
        hist_abs_res.append(abs_res.item())
        hist_rel_res.append(rel_res.item())
        hist_time.append(time.time() - tic)
        iters = 0
        quit_iter = False

        # Outer loop
        while 1:

            # Restart cycle
            V[:,0] = r / beta
            g[0] = beta
            for j in range(restart):
                if M is not None:
                    Z[:,j] = M.apply(V[:,j])
                else:
                    Z[:,j] = V[:,j]
                w = A @ Z[:,j]
                for k in range(j+1):
                    H[k,j] = torch.dot(V[:,k], w)
                    w = w - H[k,j] * V[:,k]
                H[j+1,j] = torch.linalg.norm(w)
                V[:,j+1] = w / H[j+1,j]

                # Solve min || H * y - beta * e1 ||: Givens rotation
                for k in range(j):
                    tmp      =  c[k] * H[k,j] + s[k] * H[k+1,j]
                    H[k+1,j] = -s[k] * H[k,j] + c[k] * H[k+1,j]
                    H[k,j] = tmp
                t = torch.sqrt( H[j,j]**2 + H[j+1,j]**2 )
                c[j], s[j] = H[j,j]/t, H[j+1,j]/t
                H[j,j] = c[j] * H[j,j] + s[j] * H[j+1,j]
                H[j+1,j] = 0
                g[j+1] = -s[j] * g[j]
                g[j] = c[j] * g[j]
                # End solve min || H * y - beta * e1 ||: Givens rotation

                abs_res = torch.abs(g[j+1])
                rel_res = abs_res / norm_b
                hist_abs_res.append(abs_res.item())
                hist_rel_res.append(rel_res.item())
                hist_time.append(time.time() - tic)
                iters = iters + 1
                if (rel_res < rtol) or \
                   (timeout is None and iters == max_iters) or \
                   (timeout is not None and hist_time[-1] >= timeout):
                    quit_iter = True
                    break

                if progress_bar:
                    pbar.update()
            # End restart cycle

            # Solve min || H * y - beta * e1 ||: obtain solution
            y = torch.linalg.solve_triangular(H[:j+1, :j+1],
                                              g[:j+1].view(j+1,1),
                                              upper=True).view(j+1)
            # End solve min || H * y - beta * e1 ||: obtain solution

            x = x + Z[:, :j+1] @ y
            r = b - A @ x
            beta = torch.linalg.norm(r)
            if np.allclose(hist_abs_res[-1], beta.item()) is False:
                warn('Residual tracked by least squares solve is different '
                     'from the true residual. The result of GMRES should not '
                     'be trusted.')
            if quit_iter == True:
                break
            else:
                g = g.zero_()
                g[0] = beta
        # End outer loop

        if progress_bar:
            pbar.close()

        return x, iters, hist_abs_res, hist_rel_res, hist_time
            

#-----------------------------------------------------------------------------
# Arnoldi process (m steps).
class Arnoldi():

    def build(self, A, v0=None, m=100):

        n = A.shape[0]
        if v0 is None:
            v0 = torch.normal(0, 1, size=(n,), dtype=A.dtype).to(A.device)
        beta = torch.linalg.norm(v0)
        
        V = torch.zeros(n, m+1, dtype=A.dtype).to(A.device)
        H = torch.zeros(m+1, m, dtype=A.dtype).to(A.device)

        V[:,0] = v0 / beta
        for j in range(m):
            w = A @ V[:,j]
            for k in range(j+1):
                H[k,j] = torch.dot(V[:,k], w)
                w = w - H[k,j] * V[:,k]
            H[j+1,j] = torch.linalg.norm(w)
            V[:,j+1] = w / H[j+1,j]

        Vm1 = V
        barHm = H
        return Vm1, barHm
            

#-----------------------------------------------------------------------------
if __name__ == '__main__':

    # Test Arnoldi
    from GNP.problems import gen_1d_laplacian
    A = gen_1d_laplacian(1000)
    arnoldi = Arnoldi()
    Vm1, barHm = arnoldi.build(A)
    print( torch.linalg.norm( A @ Vm1[:,:-1] - Vm1 @ barHm ) )

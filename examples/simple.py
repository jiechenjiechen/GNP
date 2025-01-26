# This code tests the GNP preconditioner for solving Ax = b. The
# matrix A comes from SuiteSparse https://sparse.tamu.edu.
#
# Usage (on a GPU machine):
#
#   python simple.py --location ~/data/SuiteSparse/ssget/mat --problem VanVelzen/std1_Jac3 --num_workers 4
#
# One may specify the input matrix A by using --location and
# --problem; and the output files by using --out_path and
# --out_file_prefix. To see their default values, type
#
#   python simple.py -h
#
# or check the first part of the main() function. This code will
# generate three plots, one for the GNP training history, one for the
# convergence history of the linear solves, and one for the
# time-to-solution for the linear solves. The linear solves include
# not using a preconditioner and using GNP.
#
# The option --num_workers specifies the number of dataloader workers
# for GNP training. The default is 0 (equivalent to 1). This works the
# best for a CPU machine. For a GPU machine, use a larger number.

import os
import time
import torch
import argparse
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

from GNP.problems import *
from GNP.solver import GMRES
from GNP.precond import *
from GNP.nn import ResGCN
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse


#-----------------------------------------------------------------------------
def main():

    # Input arguments
    parser = argparse.ArgumentParser(
        description='Solving linear system Ax = b',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--location', type=str, default='~/data/SuiteSparse/ssget/mat',
        help='root path of SuiteSparse problem '
        '(default: ~/data/SuiteSparse/ssget/mat)')
    parser.add_argument(
        '--problem', type=str, default='VanVelzen/std1_Jac3',
        help='group/name from SuiteSparse '
        '(default: VanVelzen/std1_Jac3)')
    parser.add_argument(
        '--out_path', type=str, default='./dump/',
        help='path of output figures (default: ./dump/)')
    parser.add_argument(
        '--out_file_prefix', type=str,
        help='filename prefix of output figures. If argument is not set, '
        '''default is f"{args.problem.replace('/', '_')}_"''')
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='number of dataloader workers in training GNP (default: 0)')
    args = parser.parse_args()

    # Setup and parameters
    restart = 10                # restart cycle in GMRES
    max_iters = 100             # maximum number of GMRES iterations
    timeout = None              # timeout in seconds
    rtol = 1e-8                 # relative residual tolerance in GMRES
    training_data = 'x_mix'     # type of training data x
    m = 40                      # Krylov subspace dimension for training data
    num_layers = 8              # number of layers in GNP
    embed = 16                  # embedding dimension in GNP
    hidden = 32                 # hidden dimension in MLPs in GNP
    drop_rate = 0.0             # dropout rate in GNP
    disable_scale_input = False # whether disable the scaling of inputs in GNP
    dtype = torch.float32       # training precision for GNP
    batch_size = 16             # batch size in training GNP
    grad_accu_steps = 1         # gradient accumulation steps in training GNP
    epochs = 2000               # number of epochs in training GNP
    lr = 1e-3                   # learning rate in training GNP
    weight_decay = 0.0          # weight decay in training GNP
    save_model = True           # whether save model
    hide_solver_bar = False     # whether hide progress bar in linear solver
    hide_training_bar = False   # whether hide progress bar in GNP training

    # Computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load problem
    A = load_suitesparse(args.location, args.problem, device)

    # Normalize A to avoid hassles
    A = scale_A_by_spectral_radius(A)
    
    # Print problem information
    n = A.shape[0]
    print(f'\nMatrix A: name = {args.problem}, n = {n}, nnz = {A._nnz()}')
        
    # Right-hand side b
    x = gen_x_all_ones(n).to(device)
    b = A @ x
    del x

    # Output path and filename
    args.out_path = os.path.abspath(os.path.expanduser(args.out_path))
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    if args.out_file_prefix is None:
        args.out_file_prefix = f"{args.problem.replace('/', '_')}_"
    out_file_prefix_with_path = os.path.join(args.out_path,
                                             args.out_file_prefix)

    # Solver
    solver = GMRES()

    # GMRES without preconditioner
    print('\nSolving linear system without preconditioner ...')
    solver.solve(     # dry run; timing is not accurate
        A, b, M=None, restart=restart, max_iters=max_iters,
        timeout=timeout, rtol=rtol, progress_bar=False)
    _, _, _, hist_rel_res, hist_time = solver.solve(
        A, b, M=None, restart=restart, max_iters=max_iters,
        timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    print(f'Done. Final relative residual = {hist_rel_res[-1]:.4e}')

    # GMRES with GNP: Train preconditioner
    print('\nTraining GNP ...')
    net = ResGCN(A, num_layers, embed, hidden, drop_rate,
                 scale_input=not disable_scale_input, dtype=dtype).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = None
    M = GNP(A, training_data, m, net, device)
    tic = time.time()
    hist_loss, best_loss, best_epoch, model_file = M.train(
        batch_size, grad_accu_steps, epochs, optimizer, scheduler,
        num_workers=args.num_workers,
        checkpoint_prefix_with_path=\
        out_file_prefix_with_path if save_model else None,
        progress_bar=not hide_training_bar)
    print(f'Done. Training time: {time.time()-tic} seconds')
    print(f'Loss: inital = {hist_loss[0]}, '
          f'final = {hist_loss[-1]}, '
          f'best = {best_loss}, epoch = {best_epoch}')
    if save_model:
        print(f'Best model saved in {model_file}')

    # Investigate training history of the preconditioner
    print('\nPlotting training history ...')
    plt.figure(1)
    plt.semilogy(hist_loss, label='train')
    plt.title(f'{args.problem}: Preconditioner convergence (MAE loss)')
    plt.legend()
    # plt.show()
    full_path = out_file_prefix_with_path + 'training.png'
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')

    # Load the best checkpoint
    if model_file:
        print(f'\nLoading model from {model_file} ...')
        net.load_state_dict(torch.load(model_file, map_location=device))
        M = GNP(A, training_data, m, net, device)
        print('Done.')
    else:
        print('\nNo checkpoint is saved. Use model from the last epoch.')
            
    # GMRES with GNP: Linear solve
    print('\nSolving linear system with GNP ...')
    warnings.filterwarnings('error')
    try:
        _, _, _, hist_rel_res_gnp, hist_time_gnp = solver.solve(
            A, b, M=M, restart=restart, max_iters=max_iters,
            timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    except UserWarning as w:
        print('Warning:', w)
        print('GMRES preconditioned by GNP fails')
        hist_rel_res_gnp = None
        hist_time_gnp = None
    else:
        print(f'Done. '
              f'Final relative residual = {hist_rel_res_gnp[-1]:.4e}')
    warnings.resetwarnings()

    # Investigate solution history
    print('\nPlotting solution history ...')
    plt.figure(2)
    plt.semilogy(hist_rel_res, color='C0', label='no precond')
    if hist_rel_res_gnp is not None:
        plt.semilogy(hist_rel_res_gnp, color='C7', label='GNP')
    solver_name = solver.__class__.__name__
    plt.title(f'{args.problem}: {solver_name} convergence (relative residual)')
    plt.xlabel('(Outer) Iterations')
    plt.legend()
    # plt.show()
    full_path = out_file_prefix_with_path + 'solver.png'
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')
    
    # Compare solution speed
    print('\nPlotting solution history (time to solution) ...')
    plt.figure(3)
    plt.semilogy(hist_time, hist_rel_res, color='C0', label='no precond')
    if hist_rel_res_gnp is not None:
        plt.semilogy(hist_time_gnp, hist_rel_res_gnp, color='C7', label='GNP')
    solver_name = solver.__class__.__name__
    plt.title(f'{args.problem}: {solver_name} convergence (relative residual)')
    plt.xlabel('Time (seconds)')
    plt.legend()
    # plt.show()
    full_path = out_file_prefix_with_path + 'time.png'
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')
    

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

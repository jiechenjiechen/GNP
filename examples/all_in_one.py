# This code tests the GNP package for solving Ax = b.
#
# Example usage (on a GPU machine):
#
#   python all_in_one.py --location ~/data/SuiteSparse/ssget/mat --problem VanVelzen/std1_Jac3 --preconditioners gnp gmres ilu amg jacobi --epochs 2000 --num_workers 4
#
# The code currently supports the following scenarios:
#
#   1. GMRES without preconditioner
#   2. FGMRES (GMRES preconditioned by GMRES)
#   3. GMRES preconditioned by ILU/AMG/AMG_AIR/Jacobi/BlockJacobi/GNP
#
# By default, only scenario 1 is computed. To compute other scenarios,
# use the --precondtioners option. Type
#
#   python all_in_one.py -h
#
# for detailed use of the code.
#
# TODO:
#
#   - Support mps
#   - Support float16 and bfloat16
#   - Support RHS from Suitesparse problem

import os
import json
import time
import torch
import inspect
import argparse
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

import GNP.problems as syn_problems
from GNP.problems import *
from GNP.solver import GMRES
from GNP.precond import *
from GNP.nn import ResGCN
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse


#-----------------------------------------------------------------------------
def main():

    # List of synthetic problems
    problems = [problem for problem in dir(syn_problems)
                if inspect.isfunction(getattr(syn_problems, problem))]
    problems.remove('gen_b_all_ones')
    problems.remove('gen_x_all_ones')    
    problems.remove('gen_b_randn')
    problems.remove('gen_x_randn')    
    problems = [problem[4:] for problem in problems]
    
    # Input arguments
    parser = argparse.ArgumentParser(
        description='Solving linear system Ax = b',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='available synthetic problems:\n  ' + '\n  '.join(problems))

    # Problem and right-hand side
    parser.add_argument(
        '--problem', type=str, default='1d_laplacian',
        help='name of synthetic problem (see below) or group/name from '
        'SuiteSparse (see https://sparse.tamu.edu) (default: 1d_laplacian)')
    parser.add_argument(
        '-n', type=int, default=1024,
        help='size of synthetic problem (default: 1024)')
    parser.add_argument(
        '--location', type=str, default='~/data/SuiteSparse/ssget/mat',
        help='root path of SuiteSparse problem '
        '(default: ~/data/SuiteSparse/ssget/mat)')
    parser.add_argument(
        '--rhs', choices=['b_all_ones', 'x_all_ones', 'b_randn', 'x_randn',
                          'from_problem'],
        default='x_all_ones',
        help='type of right-hand side (default: x_all_ones)')

    # Linear solver (GMRES)
    parser.add_argument(
        '--restart', type=int, default=10,
        help='restart cycle in GMRES (default: 10)')
    parser.add_argument(
        '--max_iters', type=int, default=100,
        help='maximum number of GMRES iterations (default: 100)')
    parser.add_argument(
        '--timeout', type=float, default=None,
        help='timeout in seconds (default: None). If timeout is not None, '
        'max_iters is disabled.')
    parser.add_argument(
        '--rtol', type=float, default=1e-8,
        help='relative residual tolerance in GMRES (default: 1e-8)')

    # Preconditioner choices
    parser.add_argument(
        '--preconditioners', choices=['gmres', 'gnp', 'ilu', 'amg', 'amg_air',
                                      'jacobi', 'bjacobi'],
        nargs='+',
        help='run the code with preconditioner(s)')

    # GMRES as preconditioner
    parser.add_argument(
        '--inner_iters', type=int, default=10,
        help='number of GMRES iterations as preconditioner (default: 10)')
    parser.add_argument(
        '--inner_rtol', type=float, default=1e-6,
        help='relative residual tolerance in inner GMRES solve (default: 1e-6)')
    
    # Generating streaming training data for GNP
    parser.add_argument(
        '--training_data', choices=['x_normal', 'x_subspace', 'x_mix', 'no_x'],
        default='x_mix',
        help='type of training data x (default: x_mix)')
    parser.add_argument(
        '-m', type=int, default=40,
        help='Krylov subspace dimension (default: 40). Only effective when '
        'training_data is x_subspace or x_mix')
    
    # GNP
    parser.add_argument(
        '--num_layers', type=int, default=8,
        help='number of layers in GNP (default: 8)')
    parser.add_argument(
        '--embed', type=int, default=16,
        help='embedding dimension in GNP (default: 16)')
    parser.add_argument(
        '--hidden', type=int, default=32,
        help='hidden dimension in MLPs (default: 32)')
    parser.add_argument(
        '--drop_rate', type=float, default=0.0,
        help='dropout rate in GNP (default: 0.0)')
    parser.add_argument(
        '--disable_scale_input', action='store_true',
        help='disable the scaling of inputs in the neural net')
    parser.add_argument(
        '--precision', choices=['float32', 'float16', 'bfloat16'],
        default='float32',
        help='training precision for neural net (default: float32)')

    # Training GNP
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size in training GNP (default: 16)')
    parser.add_argument(
        '--grad_accu_steps', type=int, default=1,
        help='gradient accumulation steps in training GNP (default: 1)')
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help='number of epochs in training GNP (default: 1000)')
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='learning rate in training GNP (default: 1e-3)')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='weight decay in training GNP (default: 0.0)')
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='number of dataloader workers in training GNP (default: 0)')

    # GNP model saving/loading
    parser.add_argument(
        '--model_file', type=str,
        help='if invoked, supply filename (with path) of the model used by GNP')
    parser.add_argument(
        '--save_model', action='store_true',
        help='save model (only effective when model_file is not provided)')

    # ILU preconditioner
    parser.add_argument(
        '--ilu_factors_file', type=str,
        help='if invoked, supply filename (with path) of the ilu factors '
        'used by the ILU preconditioner')
    parser.add_argument(
        '--save_ilu_factors', action='store_true',
        help='save ilu factors (only effective when ilu_factors_file '
        'is not provided)')
    parser.add_argument(
        '--fill_factor', type=float, default=None,
        help='fill factor used by scipy.sparse.linalg.spilu (default: None; '
        'i.e., use scipy default)')
    
    # Block Jacobi preconditioner
    parser.add_argument(
        '--bjac_lu_factors_file', type=str,
        help='if invoked, supply filename (with path) of the block Jacobi '
        'LU factors')
    parser.add_argument(
        '--save_bjac_lu_factors', action='store_true',
        help='save block Jacobi LU factors (only effective when '
        'bjac_lu_factors_file is not provided)')
    parser.add_argument(
        '--block_size', type=int, default=32,
        help='block size in block Jacobi preconditioner (default: 32)')
    
    # Output, progress bar, debug, etc
    parser.add_argument(
        '--out_path', type=str, default='./dump/',
        help='path of output figures (default: ./dump/)')
    parser.add_argument(
        '--out_file_prefix', type=str,
        help='filename prefix of output figures. If argument is not set, '
        '''default is f"{args.problem.replace('/', '_')}_{args.rhs}_"''')
    parser.add_argument(
        '--hide_solver_bar', action='store_true',
        help='hide progress bar in linear solver')
    parser.add_argument(
        '--hide_training_bar', action='store_true',
        help='hide progress bar in neural net training')
    
    # Print input arguments
    args = parser.parse_args()
    print('Input options. '
          '"n" and "location" may be adjusted according to "problem"')
    print(json.dumps(vars(args), indent=2))

    # Computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # - Disable mps for now. mps does not support sparse tensors, nor float64
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Matrix A. 
    #
    if args.problem in problems:
        # Synthetic problem
        func = getattr(syn_problems, 'gen_' + args.problem)
        A = func(args.n).to(device)
    else:
        # SuiteSparse problem
        A = load_suitesparse(args.location, args.problem, device)
    #
    # Normalize A before everything, to avoid hassles
    A = scale_A_by_spectral_radius(A)
    #
    n = A.shape[0]   # no guarantee that n == args.n
    print(f'\nMatrix A: name = {args.problem}, n = {n}, nnz = ' +
          (f'{A._nnz()}' if A.layout == torch.sparse_csc \
           else f'{torch.count_nonzero(A)}'))
        
    # Right-hand side b
    if args.rhs == 'b_all_ones':
        b = gen_b_all_ones(n).to(device)
    elif args.rhs == 'x_all_ones':
        x = gen_x_all_ones(n).to(device)
        b = A @ x
        del x
    elif args.rhs == 'b_randn':
        b = gen_b_randn(n).to(device)
    elif args.rhs == 'x_randn':
        x = gen_x_randn(n).to(device)
        b = A @ x
        del x
    elif args.rhs == 'from_problem':
        raise Exception(f'Right-hand side type {args.rhs} not implemented yet!')
    else:
        raise Exception(f'Unsupported right-hand side type {args.rhs}!')

    # Output path and filename
    args.out_path = os.path.abspath(os.path.expanduser(args.out_path))
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    if args.out_file_prefix is None:
        args.out_file_prefix = f"{args.problem.replace('/', '_')}_{args.rhs}_"
    out_file_prefix_with_path = os.path.join(args.out_path,
                                             args.out_file_prefix)
    if args.model_file is not None:
        args.model_file = os.path.abspath(
            os.path.expanduser(args.model_file))
    if args.ilu_factors_file is not None:
        args.ilu_factors_file = os.path.abspath(
            os.path.expanduser(args.ilu_factors_file))

    # Solver
    solver = GMRES()

    # GMRES without preconditioner
    print('\nSolving linear system without preconditioner ...')
    solver.solve(     # dry run; timing is not accurate
        A, b, M=None, restart=args.restart, max_iters=args.max_iters,
        timeout=args.timeout, rtol=args.rtol, progress_bar=False)
    _, _, _, hist_rel_res, hist_time = solver.solve(
        A, b, M=None, restart=args.restart, max_iters=args.max_iters,
        timeout=args.timeout, rtol=args.rtol,
        progress_bar=not args.hide_solver_bar)
    print(f'Done. Final relative residual = {hist_rel_res[-1]:.4e}')

    # FGMRES (GMRES preconditioned by GMRES)
    if args.preconditioners is not None and 'gmres' in args.preconditioners:
        print('\nSolving linear system with GMRES preconditioner (FGMRES) ...')
        M = GMRESPreconditioner(A, inner_iters=args.inner_iters,
                                inner_rtol=args.inner_rtol)
        _, _, _, hist_rel_res_fgmres, hist_time_fgmres = solver.solve(
            A, b, M=M, restart=args.restart, max_iters=args.max_iters,
            timeout=args.timeout, rtol=args.rtol,
            progress_bar=not args.hide_solver_bar)
        print(f'Done. Final relative residual = {hist_rel_res_fgmres[-1]:.4e}')

    # GMRES with ILU preconditioner
    if args.preconditioners is not None and 'ilu' in args.preconditioners:
        print('\nSolving linear system with ILU preconditioner ...')
        tic = time.time()
        try:
            if args.ilu_factors_file is not None: # Load ILU factors from file
                M = ILU(A, args.ilu_factors_file, False)
            else:
                full_path = out_file_prefix_with_path + 'ilu.pkl'
                if args.save_ilu_factors is True: # Compute and save to file
                    M = ILU(A, full_path, True, fill_factor=args.fill_factor)
                else:                             # Compute but do not save
                    M = ILU(A, None, False, fill_factor=args.fill_factor)
        except RuntimeError as e:
            print('RuntimeError:', e)
            print('Fail to perform ILU factorization')
            hist_rel_res_ilu = None
            hist_time_ilu = None
        else:
            if args.ilu_factors_file is not None:
                print(f'Loaded ILU factors from {args.ilu_factors_file}')
            else:
                print(f'ILU factorization time: {time.time()-tic} seconds')
                if args.save_ilu_factors is True:
                    print(f'ILU factors saved in {full_path}')
            _, _, _, hist_rel_res_ilu, hist_time_ilu = solver.solve(
                A, b, M=M, restart=args.restart, max_iters=args.max_iters,
                timeout=args.timeout, rtol=args.rtol,
                progress_bar=not args.hide_solver_bar)
            print(f'Done. Final relative residual = {hist_rel_res_ilu[-1]:.4e}')

    # GMRES with AMG preconditioner
    if args.preconditioners is not None and 'amg' in args.preconditioners:
        print('\nSolving linear system with AMG preconditioner ...')
        try:
            tic = time.time()
            M = AMGPreconditioner(A)
        except TypeError as e:
            print('TypeError:', e)
            print('Fail to construct AMG preconditioner')
            hist_rel_res_amg = None
            hist_time_amg = None
        else:
            print(f'AMG construction time: {time.time()-tic} seconds')
            _, _, _, hist_rel_res_amg, hist_time_amg = solver.solve(
                A, b, M=M, restart=args.restart, max_iters=args.max_iters,
                timeout=args.timeout, rtol=args.rtol,
                progress_bar=not args.hide_solver_bar)
            print(f'Done. Final relative residual = {hist_rel_res_amg[-1]:.4e}')

    # GMRES with AMG_AIR preconditioner
    if args.preconditioners is not None and 'amg_air' in args.preconditioners:
        print('\nSolving linear system with AMG_AIR preconditioner ...')
        try:
            tic = time.time()
            M = AMGPreconditioner_AIR(A)
        except TypeError as e:
            print('TypeError:', e)
            print('Fail to construct AMG_AIR preconditioner')
            hist_rel_res_amg_air = None
            hist_time_amg_air = None
        else:
            print(f'AMG_AIR construction time: {time.time()-tic} seconds')
            _, _, _, hist_rel_res_amg_air, hist_time_amg_air = solver.solve(
                A, b, M=M, restart=args.restart, max_iters=args.max_iters,
                timeout=args.timeout, rtol=args.rtol,
                progress_bar=not args.hide_solver_bar)
            print(f'Done. Final relative residual = '
                  f'{hist_rel_res_amg_air[-1]:.4e}')

    # GMRES with Jacobi preconditioner
    if args.preconditioners is not None and 'jacobi' in args.preconditioners:
        print('\nSolving linear system with Jacobi preconditioner ...')
        M = Jacobi(A)
        _, _, _, hist_rel_res_jacobi, hist_time_jacobi = solver.solve(
            A, b, M=M, restart=args.restart, max_iters=args.max_iters,
            timeout=args.timeout, rtol=args.rtol,
            progress_bar=not args.hide_solver_bar)
        print(f'Done. Final relative residual = {hist_rel_res_jacobi[-1]:.4e}')

    # GMRES with block Jacobi preconditioner
    if args.preconditioners is not None and 'bjacobi' in args.preconditioners:
        print('\nSolving linear system with block Jacobi preconditioner ...')
        tic = time.time()
        try:
            if args.bjac_lu_factors_file is not None: # Load factors from file
                M = BlockJacobi(A, args.bjac_lu_factors_file, False)
            else:
                full_path = out_file_prefix_with_path + 'bjac_lu.pkl'
                if args.save_bjac_lu_factors is True: # Compute and save to file
                    M = BlockJacobi(A, full_path, True,
                                    block_size=args.block_size)
                else:                                 # Compute but do not save
                    M = BlockJacobi(A, None, False, block_size=args.block_size)
        except RuntimeError as e:
            print('RuntimeError:', e)
            print('Fail to perform block Jacobi LU factorization')
            hist_rel_res_bjac_lu = None
            hist_time_bjac_lu = None
        else:
            if args.bjac_lu_factors_file is not None:
                print(f'Loaded block Jacobi LU factors from '
                      f'{args.bjac_lu_factors_file}')
            else:
                print(f'Block Jacobi LU factorization time: '
                      f'{time.time()-tic} seconds')
                if args.save_bjac_lu_factors is True:
                    print(f'Block Jacobi LU factors saved in {full_path}')
            _, _, _, hist_rel_res_bjac_lu, hist_time_bjac_lu = solver.solve(
                A, b, M=M, restart=args.restart, max_iters=args.max_iters,
                timeout=args.timeout, rtol=args.rtol,
                progress_bar=not args.hide_solver_bar)
            print(f'Done. Final relative residual = '
                  f'{hist_rel_res_bjac_lu[-1]:.4e}')
            
    # GMRES with GNP
    if args.preconditioners is not None and 'gnp' in args.preconditioners:
        
        # Training precision
        if args.precision == 'float32':
            dtype = torch.float32
        elif args.precision == 'float16' or args.precision == 'bfloat16':
            raise Exception(f'Precision {args.precision} not implemented yet!')
        else:
            raise Exception(f'Unsupported training precision {args.precision}!')

        net = ResGCN(A, args.num_layers, args.embed, args.hidden,
                     args.drop_rate,
                     scale_input=not args.disable_scale_input,
                     dtype=dtype).to(device)

        if args.model_file is None:
            
            # Optimizer
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
            scheduler = None
            
            # Train preconditioner
            print('\nTraining GNP ...')
            M = GNP(A, args.training_data, args.m, net, device)
            tic = time.time()
            hist_loss, best_loss, best_epoch, args.model_file = M.train(
                args.batch_size, args.grad_accu_steps, args.epochs, optimizer,
                scheduler, num_workers=args.num_workers,
                checkpoint_prefix_with_path=\
                out_file_prefix_with_path if args.save_model else None,
                progress_bar=not args.hide_training_bar)
            print(f'Done. Training time: {time.time()-tic} seconds')
            print(f'Loss: inital = {hist_loss[0]}, '
                  f'final = {hist_loss[-1]}, '
                  f'best = {best_loss}, epoch = {best_epoch}')
            if args.save_model:
                print(f'Best model saved in {args.model_file}')

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

        if args.model_file:
            
            # Load model for the preconditioner (either the best
            # trained model or the saved model)
            print(f'\nLoading model from {args.model_file} ...')
            net.load_state_dict(torch.load(args.model_file,
                                           map_location=device))
            M = GNP(A, args.training_data, args.m, net, device)
            print('Done.')
            
        else:
            
            # Use the model in the last training epoch
            print('\nNo checkpoint is saved. Use model from the last epoch.')

        # Solve
        print('\nSolving linear system with GNP ...')
        warnings.filterwarnings('error')
        try:
            _, _, _, hist_rel_res_gnp, hist_time_gnp = solver.solve(
                A, b, M=M, restart=args.restart, max_iters=args.max_iters,
                timeout=args.timeout, rtol=args.rtol,
                progress_bar=not args.hide_solver_bar)
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
    if args.preconditioners is not None:
        if 'gmres' in args.preconditioners:
            plt.semilogy(hist_rel_res_fgmres,
                         color='C1', label="precond'ed by GMRES")
        if 'ilu' in args.preconditioners and hist_rel_res_ilu is not None:
            plt.semilogy(hist_rel_res_ilu,
                         color='C2', label='ILU')
        if 'amg' in args.preconditioners and hist_rel_res_amg is not None:
            plt.semilogy(hist_rel_res_amg,
                         color='C3', label='AMG')
        if 'amg_air' in args.preconditioners and \
           hist_rel_res_amg_air is not None:
            plt.semilogy(hist_rel_res_amg_air,
                         color='C4', label='AMG_AIR')
        if 'jacobi' in args.preconditioners and hist_rel_res_jacobi is not None:
            plt.semilogy(hist_rel_res_jacobi,
                         color='C5', label='Jacobi')
        if 'bjacobi' in args.preconditioners and \
           hist_rel_res_bjac_lu is not None:
            plt.semilogy(hist_rel_res_bjac_lu,
                         color='C6', label='Block Jacobi')
        if 'gnp' in args.preconditioners and hist_rel_res_gnp is not None:
            plt.semilogy(hist_rel_res_gnp,
                         color='C7', label='GNP')
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
    if args.preconditioners is not None:
        if 'gmres' in args.preconditioners:
            plt.semilogy(hist_time_fgmres, hist_rel_res_fgmres,
                         color='C1', label="precond'ed by GMRES")
        if 'ilu' in args.preconditioners and hist_rel_res_ilu is not None:
            plt.semilogy(hist_time_ilu, hist_rel_res_ilu,
                         color='C2', label='ILU')
        if 'amg' in args.preconditioners and hist_rel_res_amg is not None:
            plt.semilogy(hist_time_amg, hist_rel_res_amg,
                         color='C3', label='AMG')
        if 'amg_air' in args.preconditioners and \
           hist_rel_res_amg_air is not None:
            plt.semilogy(hist_time_amg_air, hist_rel_res_amg_air,
                         color='C4', label='AMG_AIR')
        if 'jacobi' in args.preconditioners and hist_rel_res_jacobi is not None:
            plt.semilogy(hist_time_jacobi, hist_rel_res_jacobi,
                         color='C5', label='Jacobi')
        if 'bjacobi' in args.preconditioners and \
           hist_rel_res_bjac_lu is not None:
            plt.semilogy(hist_time_bjac_lu, hist_rel_res_bjac_lu,
                         color='C6', label='Block Jacobi')
        if 'gnp' in args.preconditioners and hist_rel_res_gnp is not None:
            plt.semilogy(hist_time_gnp, hist_rel_res_gnp,
                         color='C7', label='GNP')
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

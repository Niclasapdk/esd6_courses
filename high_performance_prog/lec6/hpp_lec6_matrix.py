#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

def matmul_sequential(A, B):
    """
    Naive O(N^3) matrix multiplication: C = A * B
    (A, B are N x N, returns N x N)
    """
    N = A.shape[0]
    C = np.zeros((N, N), dtype=A.dtype)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Matrix dimension
    N = 50  # Adjust as needed; must be divisible by 'size' for this example
    # e.g., if you run with mpiexec -n 5, ensure 50 % 5 == 0

    # On rank 0, create random NxN matrices
    if rank == 0:
        # Check divisibility
        if N % size != 0:
            raise ValueError(f"N={N} must be divisible by size={size} for simple row-scatter.")
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)
    else:
        A = None
        B = None

    # --------------------------------------------------------------------------
    # 1) Sequential multiplication (only on rank 0)
    # --------------------------------------------------------------------------
    if rank == 0:
        seq_start = MPI.Wtime()
        C_seq = matmul_sequential(A, B)
        seq_end = MPI.Wtime()
        seq_time = seq_end - seq_start
    else:
        C_seq = None
        seq_time = None

    # --------------------------------------------------------------------------
    # 2) Parallel multiplication using MPI
    # --------------------------------------------------------------------------
    comm.Barrier()  # synchronize before timing
    par_start = MPI.Wtime()

    # Broadcast N so all ranks know the matrix size
    N = comm.bcast(N, root=0)

    # Each rank will handle (N/size) rows of A
    rows_per_proc = N // size

    # Allocate local storage for a chunk of A
    local_A = np.zeros((rows_per_proc, N), dtype=np.float64)

    # Scatter rows of A
    if rank == 0:
        # Root passes the real array and the send-count
        comm.Scatter([A, rows_per_proc*N, MPI.DOUBLE], local_A, root=0)
    else:
        # Non-root ranks pass None + 0 count for the send buffer
        comm.Scatter([None, 0, MPI.DOUBLE], local_A, root=0)

    # Allocate B on all ranks
    if rank == 0:
        # Root already has B
        pass
    else:
        B = np.empty((N, N), dtype=np.float64)

    # Broadcast B to every rank
    comm.Bcast([B, N*N, MPI.DOUBLE], root=0)

    # Now compute local product chunk: local_C = local_A * B
    local_C = np.zeros((rows_per_proc, N), dtype=np.float64)
    for i in range(rows_per_proc):
        for j in range(N):
            for k in range(N):
                local_C[i, j] += local_A[i, k] * B[k, j]

    # Gather local_C into the full C on rank 0
    if rank == 0:
        C_par = np.zeros((N, N), dtype=np.float64)
        comm.Gather([local_C, rows_per_proc*N, MPI.DOUBLE],
                    [C_par, rows_per_proc*N, MPI.DOUBLE],
                    root=0)
    else:
        comm.Gather([local_C, rows_per_proc*N, MPI.DOUBLE],
                    [None, 0, MPI.DOUBLE],
                    root=0)
        C_par = None

    comm.Barrier()
    par_end = MPI.Wtime()
    par_time = par_end - par_start

    # --------------------------------------------------------------------------
    # Compare results and print
    # --------------------------------------------------------------------------
    if rank == 0:
        print("=== Matrix Multiplication Comparison ===")
        print(f"Matrix size: {N}x{N}")
        print(f"Number of processes: {size}\n")

        print(f"Sequential time: {seq_time:.6f} s")
        print(f"Parallel time:   {par_time:.6f} s")

        # Optional correctness check
        if not np.allclose(C_seq, C_par, atol=1e-7):
            print("WARNING: Results differ between sequential and parallel!")
        else:
            print("Check: Parallel result matches sequential result (within tolerance).")

        speedup = seq_time / par_time if par_time > 0 else 0.0
        print(f"Speedup: {speedup:.2f}x")

        # Answers to the questions:
        # a) Which one is faster? 
        #    - For large N and multiple processes, usually the MPI version is faster.
        #      For small N or fewer processes, the sequential might be faster (less overhead).
        #
        # b) Which one was easier to code?
        #    - The sequ

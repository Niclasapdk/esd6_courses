from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Number of elements and iterations
N_ELEMENTS = 10000
N_ITER = 10

# Prepare arrays
# Root will initialize data; other ranks use empty placeholders
if rank == 0:
    data_bcast = np.arange(N_ELEMENTS, dtype=np.float64)
else:
    data_bcast = np.empty(N_ELEMENTS, dtype=np.float64)

# ------------------------------------------------------------------------------
# 1. MANUAL BROADCAST (Send/Receive) TIMING
# ------------------------------------------------------------------------------
manual_bcast_times = []

for _ in range(N_ITER):
    comm.Barrier()           # Sync before timing
    t_start = MPI.Wtime()

    # Root manually sends data to each rank
    if rank == 0:
        for dest in range(1, size):
            comm.Send(data_bcast, dest=dest, tag=100)
    else:
        comm.Recv(data_bcast, source=0, tag=100)

    comm.Barrier()           # Sync after sending/receiving
    t_end = MPI.Wtime()

    manual_bcast_times.append(t_end - t_start)

# Each rank has a list of iteration times; compute local average
manual_bcast_local_avg = np.mean(manual_bcast_times)
# Reduce to get the sum of all ranks' local averages on root
manual_bcast_global_sum = comm.reduce(manual_bcast_local_avg, op=MPI.SUM, root=0)

if rank == 0:
    avg_manual_bcast_time = manual_bcast_global_sum / size
else:
    avg_manual_bcast_time = None

# ------------------------------------------------------------------------------
# 2. BUILT-IN BCAST TIMING
# ------------------------------------------------------------------------------
bcast_times = []

for _ in range(N_ITER):
    # Re-initialize the data for each iteration
    if rank == 0:
        data_bcast = np.arange(N_ELEMENTS, dtype=np.float64)
    else:
        data_bcast = np.empty(N_ELEMENTS, dtype=np.float64)

    comm.Barrier()
    t_start = MPI.Wtime()

    # Built-in broadcast
    comm.Bcast(data_bcast, root=0)

    comm.Barrier()
    t_end = MPI.Wtime()

    bcast_times.append(t_end - t_start)

bcast_local_avg = np.mean(bcast_times)
bcast_global_sum = comm.reduce(bcast_local_avg, op=MPI.SUM, root=0)

if rank == 0:
    avg_bcast_time = bcast_global_sum / size
else:
    avg_bcast_time = None

# ------------------------------------------------------------------------------
# PREPARE FOR SCATTER
# ------------------------------------------------------------------------------
chunk_size = N_ELEMENTS // size

if rank == 0:
    data_scatter = np.arange(N_ELEMENTS, dtype=np.float64)
else:
    data_scatter = None

recvbuf = np.empty(chunk_size, dtype=np.float64)

# ------------------------------------------------------------------------------
# 3. MANUAL SCATTER (Send/Receive) TIMING
# ------------------------------------------------------------------------------
manual_scatter_times = []

for _ in range(N_ITER):
    comm.Barrier()
    t_start = MPI.Wtime()

    # Manual scatter from root
    if rank == 0:
        for dest in range(size):
            start_i = dest * chunk_size
            end_i = start_i + chunk_size

            if dest == 0:
                # Copy locally for rank 0
                recvbuf[:] = data_scatter[start_i:end_i]
            else:
                comm.Send(data_scatter[start_i:end_i], dest=dest, tag=200)
    else:
        comm.Recv(recvbuf, source=0, tag=200)

    comm.Barrier()
    t_end = MPI.Wtime()

    manual_scatter_times.append(t_end - t_start)

manual_scatter_local_avg = np.mean(manual_scatter_times)
manual_scatter_global_sum = comm.reduce(manual_scatter_local_avg, op=MPI.SUM, root=0)

if rank == 0:
    avg_manual_scatter_time = manual_scatter_global_sum / size
else:
    avg_manual_scatter_time = None

# ------------------------------------------------------------------------------
# 4. BUILT-IN SCATTER TIMING
# ------------------------------------------------------------------------------
scatter_times = []

for _ in range(N_ITER):
    # Re-initialize big array on root
    if rank == 0:
        data_scatter = np.arange(N_ELEMENTS, dtype=np.float64)
    else:
        data_scatter = None

    comm.Barrier()
    t_start = MPI.Wtime()

    # Built-in scatter
    comm.Scatter(data_scatter, recvbuf, root=0)

    comm.Barrier()
    t_end = MPI.Wtime()

    scatter_times.append(t_end - t_start)

scatter_local_avg = np.mean(scatter_times)
scatter_global_sum = comm.reduce(scatter_local_avg, op=MPI.SUM, root=0)

if rank == 0:
    avg_scatter_time = scatter_global_sum / size
else:
    avg_scatter_time = None

# ------------------------------------------------------------------------------
# PRINT RESULTS (Only on Rank 0)
# ------------------------------------------------------------------------------
if rank == 0:
    print("=== Broadcast Times (Average per Process) ===")
    print(f"Manual Send/Recv Broadcast: {avg_manual_bcast_time:.6f} seconds")
    print(f"Built-in MPI Bcast:         {avg_bcast_time:.6f} seconds")

    print("\n=== Scatter Times (Average per Process) ===")
    print(f"Manual Send/Recv Scatter:   {avg_manual_scatter_time:.6f} seconds")
    print(f"Built-in MPI Scatter:       {avg_scatter_time:.6f} seconds")
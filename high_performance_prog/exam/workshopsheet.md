# Notes workshop HPP

## BFS for Large Graphs

### A.

### II.
Isuses:

* Claude says when a vertice is created an edge going to and from the newly created vertice is also created so double the amount of edges.
* Connectivity issues no logic preventing vertices with no edge

### B.

### V
Issues:
* Race conditions in mask array no control
* The numba uses all available cores cant specify like we do
    * We have fcked up with threads we just use single thread for each processor for loop with different doesnt change thread count so the table with timings is only for 8 threads and not different threads https://numba.pydata.org/numba-doc/dev/user/threading-layer.html

## C

### II
Issues:
* Frontier cost may be incorrect should look more into this doesnt take processes into account maybe it should?

### IV
Issues:
* The graph generated is shared among all ranks which essentially is a massive waste of ressources should partition the processors out so they cover only their part of the graph??.
* There exists no comparisson between theoretical and the measured timings.

## JPEG Compression Speedup Analysis - Correction

### I
Work and complexity assumes fft but we use matrix mul instead 2D DCT yields instead \theta(NMK)

### II 
Parallel DAG figure block 3 and 5 are embarrassingly parallel not partial.

But the entire JPEG algorithm is partial parallel.


### V


### VII

## **Problem**
MPI parallel JPEG compression: image divided into 8×8 blocks, distributed across P processors.

## **Original Analysis (INCORRECT)**
```
S_p = (N_blocks × t_comp + 8×8 × N_blocks × t_w) / ((N_blocks × t_comp)/P + 8×8 × N_blocks × t_w)
```

### **What's Wrong:**
1. ❌ Communication appears in numerator (sequential has no communication)
2. ❌ Communication doesn't scale with processors (missing ÷P)
3. ❌ Missing startup costs
4. ❌ Wrong data size (underestimated communication cost)

## **Corrected Analysis**

### **Sequential Time:**
```
T_seq = N_blocks × t_comp
```

### **Parallel Time:**
```
T_par = (N_blocks × t_comp)/P + startup + (H × W + 128) × t_w
```

### **Corrected Speedup:**
```
S_p = (N_blocks × t_comp) / ((N_blocks × t_comp)/P + startup + (H × W + 128) × t_w)
```

## **Why This is Correct:**

1. ✅ **No communication in sequential** - Only computation time
2. ✅ **Computation scales with P** - Each processor handles N_blocks/P blocks  
3. ✅ **Includes startup costs** - Network setup overhead
4. ✅ **Correct data transmission** - Based on actual code implementation

### **Data Transmitted (from actual code):**
- **Full image X**: H × W values (entire image sent to each process)
- **T matrix**: 8×8 = 64 values  
- **Q matrix**: 8×8 = 64 values
- **Total per process**: H × W + 128 values

**Note**: This is inefficient - each process gets entire image but only uses N_blocks/P blocks

## **Key Insight**
The actual implementation is communication-inefficient - each process receives the entire image (H×W values) but only processes N_blocks/P blocks. This creates a severe communication bottleneck that limits speedup more than optimal MPI would.

---
# Different packages description
### **Sequential Implementation**

1. **Explicit Vectorization with NumPy**

   * **Capability**: Leverages highly optimized C-based array operations.
   * **Use Case**: Ideal for numerical computations, matrix operations, and broadcasting.
   * **Performance**: Much faster than raw Python loops, but not parallel.
   * **Limitation**: Cannot optimize operations with control flow (e.g., `if` statements inside loops).

2. **Just-in-Time Compilation (JIT) with Numba**

   * **Capability**: Compiles Python functions to machine code at runtime.
   * **Use Case**: Loop-heavy numerical functions, compatible with NumPy arrays.
   * **Performance**: Drastically speeds up code without rewriting in C/C++.
   * **Limitation**: Requires statically typed, NumPy-compatible code; no parallelism unless specified.

3. **Numba Vectorization (`@vectorize`)**

   * **Capability**: Applies functions element-wise like NumPy ufuncs.
   * **Use Case**: Creating fast custom ufuncs with potential for SIMD optimizations.
   * **Performance**: Faster than loops; supports CPU and GPU backends.
   * **Limitation**: Works best for element-wise operations.

### **Parallel Implementation**

4. **Numba JIT with `parallel=True`**

   * **Capability**: Automatically parallelizes loops with `prange`.
   * **Use Case**: Loop-based numerical computations needing shared memory.
   * **Performance**: Significant boost on multi-core CPUs.
   * **Limitation**: Requires careful loop structure for effective parallelization.

5. **Multiprocessing**

   * **Capability**: Spawns multiple Python processes with separate memory.
   * **Use Case**: CPU-bound tasks that can be split into independent subtasks.
   * **Performance**: Bypasses the Global Interpreter Lock (GIL).
   * **Limitation**: Overhead of inter-process communication; not suited for shared memory.

6. **Threading**

   * **Capability**: Runs threads within a single process (shared memory).
   * **Use Case**: I/O-bound tasks (e.g., file, network operations).
   * **Performance**: Limited by GIL for CPU-bound tasks; good for I/O parallelism.
   * **Limitation**: Not ideal for CPU-bound code due to GIL.

7. **Pool Executor (`concurrent.futures`)**

   * **Capability**: High-level interface for managing worker pools (processes or threads).
   * **Use Case**: Simplified parallelism with `map` or `submit`.
   * **Performance**: Similar to multiprocessing/threading depending on backend.
   * **Limitation**: Less fine-grained control compared to low-level APIs.

8. **Message Passing Interface (MPI)** (via `mpi4py`)

   * **Capability**: Distributed computing using inter-process communication over network.
   * **Use Case**: HPC, cluster computing, large-scale simulations.
   * **Performance**: Scales across multiple machines and cores.
   * **Limitation**: Complex setup; requires cluster environment and knowledge of MPI.

9. **GPU with CUDA or OpenCL** (via Numba/CuPy/PyOpenCL)

   * **Capability**: Offloads computation to massively parallel GPU cores.
   * **Use Case**: Large-scale parallel operations, deep learning, simulations.
   * **Performance**: Orders of magnitude faster for highly parallel workloads.
   * **Limitation**: Requires hardware and setup; more complex programming model.

---
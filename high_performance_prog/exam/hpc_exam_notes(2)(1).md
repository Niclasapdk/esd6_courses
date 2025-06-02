# COMTEK6/ESD 6: High Performance Programming - Exam Notes

## **Lecture 1: Introduction to Parallel Processing**

### **Types of Parallelism**
- **Bit-level**: Vector operations (SIMD)
- **Instruction-level**: CPU pipeline, superscalar
- **Statement-level**: Independent operations
- **Task-level**: Separate processes/threads

**Parallel Categories**:
- **Task Parallelism**: Different tasks on different data
- **Data Parallelism**: Same task on different data parts
- **Pipeline Parallelism**: Sequential stages, each processing different data

### **Flynn's Taxonomy**
- **SISD**: Single Instruction, Single Data (sequential)
- **SIMD**: Single Instruction, Multiple Data (vectorization)
- **MISD**: Multiple Instruction, Single Data (rare)
- **MIMD**: Multiple Instruction, Multiple Data (most parallel systems)

### **Programming Models**
- **Shared Memory**: Threads share address space (OpenMP)
- **Distributed Memory**: Separate address spaces, message passing (MPI)
- **Hybrid**: Combination (MPI + OpenMP)

### **Performance Metrics**
- **Speedup**: S_p = T_1 / T_p
- **Efficiency**: E_p = S_p / p = T_1 / (p × T_p)
- **Cost**: C_p = p × T_p (should ≈ T_1 for cost-optimal)

### **Amdahl's Law**
S_p = 1 / (f + (1-f)/p)
- f = fraction of sequential code
- Maximum speedup limited by sequential portion
- Example: f=0.1 → max speedup = 10

---

## **Lecture 2: Communications and Data Structures**

### **Network Topologies**
| Topology | Diameter | Degree | Bisection Width |
|----------|----------|--------|-----------------|
| Chain | n-1 | 2 | 1 |
| Ring | ⌊n/2⌋ | 2 | 2 |
| Mesh (k×k) | 2(k-1) | 4 | k |
| Hypercube (d-dim) | d | d | 2^(d-1) |

### **Communication Costs by Topology**

**Chain Topology:**
- Point-to-point: (distance) × (startup cost + m × tw)
- Broadcast: (n-1) × (startup cost + m × tw) - linear propagation
- All-to-all: O(n²) × (startup cost + m × tw)

**Ring Topology:**
- Point-to-point: min(distance, n-distance) × (startup cost + m × tw)
- Broadcast: ⌊n/2⌋ × (startup cost + m × tw) - bidirectional
- All-to-all: (n-1) × (startup cost + m × tw) with pipelining

**Mesh Topology (k×k):**
- Point-to-point: (|x₁-x₂| + |y₁-y₂|) × (startup cost + m × tw)
- Broadcast: 2(k-1) × (startup cost + m × tw) - tree along rows/cols
- All-to-all: 2(k-1) × (startup cost + m × tw) per phase

**Hypercube Topology:**
- Point-to-point: (Hamming distance) × (startup cost + m × tw)
- Broadcast: d × (startup cost + m × tw) - recursive doubling
- All-to-all: d × (startup cost + m × tw) - optimal for collective ops

**Recursive Doubling (any topology):**
- Broadcast/Reduce: ⌈log₂ p⌉ × (startup cost + m × tw)

### **Communication Patterns**
- **One-to-All Broadcast**: ⌈log₂ p⌉ × (startup cost + m × tw)
- **All-to-One Reduction**: ⌈log₂ p⌉ × (startup cost + m × tw)
- **All-to-All**: Each process sends to every other
- **Scatter**: Distribute different data to each process
- **Gather**: Collect different data from each process

**Communication Cost**: T_comm = startup cost + m × tw
- startup cost = latency (α)
- m = message size (words)
- tw = per word transfer time (= 1/bandwidth)
- Alternative notation: T_comm = α + m/β where β = bandwidth

---

## **Lecture 3: Paradigms and Design Process**

### **Parallelizability Spectrum**
1. **Embarrassingly Parallel**: No dependencies (Monte Carlo)
2. **Regular**: Simple dependencies (matrix operations)
3. **Irregular**: Complex dependencies (graph algorithms)
4. **Non-parallelizable**: Sequential dependencies

### **Factors Affecting Parallelizability**
- **Task Independence**: Fewer dependencies = better parallelism
- **Data Dependencies**: RAW, WAR, WAW limit parallelism
- **Granularity**: Task size vs overhead trade-off
- **Synchronization Cost**: Communication and waiting time

### **Design Paradigms**
- **Divide and Conquer**: Recursively split problem
- **Binary Tree**: Hierarchical reduction/broadcast
- **Growing by Doubling**: Recursive doubling pattern

### **Dependency Analysis**
- **Control Flow Graph**: Shows task dependencies
- **Data Flow Graph**: Shows data dependencies
- **Critical Path**: Longest dependency chain
- **Critical Path Length (CPL)**: Lower bound on parallel time

### **Concurrency vs Performance**
- **Degree of Concurrency**: Max tasks that can run simultaneously
- **Granularity vs Overhead**: Finer grain = more overhead, better load balance

**Granularity Trade-off**:
- **Fine-grained**: Small tasks, high overhead, better load balance
- **Coarse-grained**: Large tasks, low overhead, potential load imbalance
- **Optimal**: Task execution time >> task creation time

---

## **Lecture 4: Analytical Modeling - Detailed Algorithm Analysis**

### **Asymptotic Analysis**
- **T_parallel(n,p) = T_comp(n,p) + T_comm(n,p) + T_sync(p)**
- Express runtime as function of input size (n) and processors (p)

### **Scalability Laws**
**Amdahl's Law** (fixed problem size):
S_p = 1 / (f + (1-f)/p)

**Gustafson's Law** (scaled problem size):
S_p = p - f(p-1) = f + p(1-f)

### **Parallel Reduce Algorithm Analysis**

**Problem**: Combine n elements using associative operation (sum, max, etc.)

**Sequential Reduce**:
- Time: O(n)
- Work: n-1 operations

**Parallel Reduce (Tree-based)**:
```
Step 1: [a,b,c,d,e,f,g,h] → [a⊕b, c⊕d, e⊕f, g⊕h] (4 ops parallel)
Step 2: [ab, cd, ef, gh] → [ab⊕cd, ef⊕gh] (2 ops parallel)
Step 3: [abcd, efgh] → [abcd⊕efgh] (1 op)
```
- **Time**: O(log n) with n/2 processors
- **Work**: n-1 operations (same as sequential)
- **Speedup**: O(n/log n)
- **Efficiency**: O(1/log n)
- **Communication**: O(log p) messages in tree

### **Parallel Prefix Scan (Detailed Analysis)**

**Problem**: Given [a₁, a₂, ..., aₙ], compute [a₁, a₁⊕a₂, a₁⊕a₂⊕a₃, ...]

**Sequential Scan**:
```
result[0] = a[0]
for i = 1 to n-1:
    result[i] = result[i-1] ⊕ a[i]
```
- **Time**: O(n)
- **Work**: n-1 operations

**Naive Parallel Scan**:
```
For each position i in parallel:
    result[i] = a[0] ⊕ a[1] ⊕ ... ⊕ a[i]
```
- **Time**: O(log n) with n² processors
- **Work**: O(n²) - NOT work-efficient!

**Work-Efficient Parallel Scan (Two-Phase)**:

**Phase 1: Up-sweep (Build reduction tree)**
```
d = 0: [1,2,3,4,5,6,7,8] → [_,3,_,7,_,11,_,15]
d = 1: [_,3,_,7,_,11,_,15] → [_,_,_,10,_,_,_,26]
d = 2: [_,_,_,10,_,_,_,26] → [_,_,_,_,_,_,_,36]
```

**Phase 2: Down-sweep (Distribute partial sums)**
```
d = 2: [_,_,_,_,_,_,_,36] → [_,_,_,10,_,_,_,0]
d = 1: [_,_,_,10,_,_,_,0] → [_,3,_,0,_,11,_,10]
d = 0: [_,3,_,0,_,11,_,10] → [0,1,3,3,7,7,11,11]
```

**Analysis**:
- **Time**: O(log n)
- **Work**: 2(n-1) operations - Work-efficient!
- **Processors**: O(n/log n) for optimal efficiency

### **Applications of Parallel Prefix**

**1. Polynomial Evaluation**: P(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₀
- Use prefix to compute [x⁰, x¹, x², ..., xⁿ]
- Multiply with coefficients and reduce

**2. Linear Recurrences**: xᵢ = aᵢxᵢ₋₁ + bᵢ
- Transform to matrix operations
- Use associative matrix multiplication in prefix

**3. Parallel Binary Search**:
- Sort multiple queries simultaneously
- Use prefix operations to coordinate searches

### **Cost Optimality**
Algorithm is cost-optimal if: **C_p = p × T_p ≤ c × T_sequential**

Examples:
- Parallel reduce: Cost-optimal (C_p = n)
- Naive prefix scan: NOT cost-optimal (C_p = n²)
- Work-efficient prefix: Cost-optimal (C_p = 2n)

---

## **Lecture 5: Practical Aspects - Hardware and Vectorization**

### **Hardware Platforms**
**CPU vs GPU Architecture**:
- **CPU**: Few (4-16) complex cores, optimized for latency, good for sequential
- **GPU**: Many (1000s) simple cores, optimized for throughput, good for parallel

**Memory Hierarchy**: Registers → L1 Cache → L2 Cache → L3 Cache → RAM → Disk
- Access times: 1 cycle → 3-5 cycles → 10-20 cycles → 50-100 cycles → 100-300 cycles → millions of cycles

### **SIMD (Single Instruction, Multiple Data) Concepts**

**SIMD Hardware**:
- **SSE**: 128-bit registers → 4 float32 or 2 float64 operations
- **AVX**: 256-bit registers → 8 float32 or 4 float64 operations
- **AVX-512**: 512-bit registers → 16 float32 or 8 float64 operations

**Vectorization Requirements**:
- **No loop-carried dependencies**: Each iteration independent
- **Regular memory access**: Sequential or strided patterns
- **Minimal branching**: All elements follow same control flow
- **Compatible data types**: Same type and size

### **Data Dependencies (Vectorization Blockers)**

**Types of Dependencies**:
- **RAW (Read After Write)**: True dependency
  ```
  a[i] = x[i] + y[i]  // Write
  b[i] = a[i] * 2     // Read - DEPENDENCY!
  ```
- **WAR (Write After Read)**: Anti-dependency
- **WAW (Write After Write)**: Output dependency

**Loop-Carried Dependencies**:
```
// BAD - Cannot vectorize
for i = 1 to n:
    a[i] = a[i-1] + b[i]  // Depends on previous iteration

// GOOD - Can vectorize  
for i = 0 to n:
    a[i] = b[i] + c[i]    // No dependencies
```

### **Loop Unrolling**
**Purpose**: Reduce loop overhead, improve instruction-level parallelism

**Before unrolling**:
```
for i = 0 to n:
    result[i] = a[i] * 2 + b[i]
    // Loop overhead: increment, compare, branch
```

**After unrolling (factor 4)**:
```
for i = 0 to n step 4:
    result[i]   = a[i]   * 2 + b[i]
    result[i+1] = a[i+1] * 2 + b[i+1]
    result[i+2] = a[i+2] * 2 + b[i+2]
    result[i+3] = a[i+3] * 2 + b[i+3]
    // Same loop overhead, 4× more work
```

### **Branching and Vectorization**

**Problem**: SIMD requires uniform control flow
```
// BAD for SIMD
for i = 0 to n:
    if condition[i]:
        result[i] = expensive_function(a[i])
    else:
        result[i] = cheap_function(a[i])
```

**Solutions**:
1. **Predication**: Compute both, select result
2. **Masked operations**: Modern SIMD supports conditional execution
3. **Branch elimination**: Mathematical transformation

### **Vectorization in Python**

**NumPy Universal Functions (ufuncs)**:
```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
result = a + b           # Vectorized addition
result = np.sin(a)       # Vectorized function
result = np.maximum(a, b) # Element-wise operations
```

**Broadcasting**:
```python
a = np.array([[1, 2, 3],    # Shape: (2, 3)
              [4, 5, 6]])
b = np.array([10, 20, 30])  # Shape: (3,)
result = a + b  # b broadcast to each row: (2, 3)
```

**Numba JIT Compilation**:
```python
from numba import jit

@jit(target='cpu')        # CPU optimization
def cpu_function(arr):
    return np.sum(arr ** 2)

@jit(target='parallel')   # Automatic parallelization  
def parallel_function(arr):
    return np.sum(arr ** 2)

@jit(target='cuda')       # GPU execution
def gpu_function(arr):
    return np.sum(arr ** 2)
```

**Performance Impact of Broadcasting and Type Casting**:
- **Same types**: Direct vectorization, optimal performance
- **Mixed types**: Requires type conversion, slower
- **Broadcasting**: May require memory copies, affects cache performance

**Amdahl's Law for Vectorization**:
If f = fraction of non-vectorizable code, vectorization speedup limited by:
S_v = 1 / (f + (1-f)/v) where v = vector width

---

## **Lecture 6: Distributed Memory Programming with MPI**

### **Distributed Memory Programming Basics**

**Distributed vs Shared Memory**:
| Aspect | Shared Memory | Distributed Memory |
|--------|---------------|-------------------|
| Memory Model | Single address space | Separate address spaces |
| Communication | Direct memory access | Explicit message passing |
| Scalability | Limited by memory bandwidth | Scales to thousands of nodes |
| Programming | Easier (implicit communication) | Harder (explicit communication) |
| Debugging | Race conditions | Communication errors |
| Fault Tolerance | Single point of failure | Process isolation |

**Challenges**:
- **Explicit communication**: Must manually send/receive data
- **Communication overhead**: Network latency and bandwidth limits
- **Synchronization**: Coordinating distributed processes

**Hybrid Architectures**: Real systems combine shared + distributed (multi-core nodes in clusters)

### **MPI Programming Model**

**Basic MPI Setup**:
```python
from mpi4py import MPI
comm = MPI.COMM_WORLD     # Communicator (all processes)
rank = comm.Get_rank()    # Process ID (0, 1, 2, ...)
size = comm.Get_size()    # Total number of processes
```

**Point-to-Point Communication**:
```python
# Blocking
if rank == 0:
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)

# Non-blocking
if rank == 0:
    req = comm.isend(data, dest=1, tag=11)
    req.wait()  # Complete later
elif rank == 1:
    req = comm.irecv(source=0, tag=11)
    data = req.wait()
```

### **Collective Communication Patterns**

**Broadcast**: One process sends same data to all others
```python
data = comm.bcast(data, root=0)  # Root sends to all
# Cost: ⌈log₂ p⌉ × (startup cost + m × tw)
```

**Scatter**: Distribute different chunks to each process
```python
local_data = comm.scatter(data, root=0)  # Distribute chunks
# Process 0 gets chunk 0, Process 1 gets chunk 1, etc.
```

**Gather**: Collect different data from each process
```python
all_data = comm.gather(local_data, root=0)  # Collect at root
# Root receives [data_from_0, data_from_1, data_from_2, ...]
```

**Reduce**: Combine data using operation (sum, max, etc.)
```python
total = comm.reduce(local_sum, op=MPI.SUM, root=0)
# Cost: ⌈log₂ p⌉ × (startup cost + m × tw)
```

**All-reduce**: Reduce + broadcast result to all
```python
total = comm.allreduce(local_sum, op=MPI.SUM)
# All processes get the total
```

### **Network Models in Real-World Supercomputers**
- **Linear/Ring**: Simple, low cost, limited scalability
- **Mesh**: 2D/3D grids, good for scientific computing
- **Hypercube**: Optimal communication, expensive to build
- **Fat Tree**: Hierarchical, used in modern clusters
- **Fully Connected**: Ideal but impractical for large systems

### **Optimized Algorithms**
**Recursive Doubling**: Reduces communication steps
- Works well on hypercube and can be mapped to other topologies
- **Broadcast**: log₂ p steps instead of p-1 linear steps
- **Reduce**: Same pattern, data flows toward root

### **Real-World Applications**
- **Parallel Matrix-Vector Multiplication**: Distribute matrix rows
- **Hyperparameter Tuning in ML**: Distribute parameter combinations
- **Cloud Computing**: Distributed microservices
- **Edge AI**: Distributed inference across edge devices
- **Satellite-based Earth Observation**: Process distributed sensor data

---

## **Lecture 7: Shared Memory Programming**

### **Shared Memory Programming Basics**

**Shared Memory Model**: All threads/processes see same memory space
- **Advantages**: Easy data sharing, low communication cost
- **Disadvantages**: Race conditions, limited scalability

**SIMD vs MIMD Models**:
- **SIMD**: Same instruction on multiple data (vectorization)
- **MIMD**: Different instructions on different data (threading)

**NUMA (Non-Uniform Memory Access)**: Memory access time varies by location
- Local memory faster than remote memory
- Important for performance on multi-socket systems

### **Synchronization Issues (The Big Four)**

**1. Race Conditions**: Multiple threads access shared data, outcome depends on timing
```python
# UNSAFE - Race condition
counter = 0
def increment():
    temp = counter     # 1. Read
    temp = temp + 1    # 2. Increment  
    counter = temp     # 3. Write (another thread can interfere!)

# SAFE - Use locks
lock = threading.Lock()
def safe_increment():
    with lock:
        global counter
        counter += 1   # Atomic operation
```

**2. Deadlock**: Circular waiting - threads wait for each other forever
```python
# Deadlock scenario:
# Thread 1: Has lock_A, wants lock_B  
# Thread 2: Has lock_B, wants lock_A
# Solution: Always acquire locks in same order
```

**3. Livelock**: Threads change state but make no progress
- Threads continuously back off and retry
- Solution: Exponential backoff with randomization

**4. Starvation**: Thread perpetually denied access to resources
- High-priority threads monopolize resources
- Solution: Fair scheduling, aging

### **Synchronization Techniques**

**Mutex Locks**: Mutual exclusion
```python
lock = threading.Lock()
with lock:  # Only one thread in critical section
    shared_resource.modify()
```

**Condition Variables**: Wait for specific conditions
```python
condition = threading.Condition()
with condition:
    while not ready:
        condition.wait()  # Wait for signal
    # Proceed when condition met
```

**Semaphores**: Limit number of concurrent accesses
```python
semaphore = threading.Semaphore(3)  # Max 3 threads
with semaphore:
    # At most 3 threads here simultaneously
    critical_work()
```

**Barriers**: Wait for all threads to reach sync point
```python
barrier = threading.Barrier(4)  # Wait for 4 threads
# All threads wait here until 4 arrive
barrier.wait()
# All proceed together
```

### **Producer-Consumer Model**
Classic pattern for thread communication:
```python
queue = Queue()

def producer():
    for item in generate_items():
        queue.put(item)  # Thread-safe
        
def consumer():
    while True:
        item = queue.get()  # Blocks until available
        process(item)
```

### **Threading vs Multiprocessing Trade-offs**

| Aspect | Threading | Multiprocessing |
|--------|-----------|-----------------|
| Memory | Shared address space | Separate address spaces |
| Creation Overhead | Low (~μs) | High (~ms) |
| Communication | Direct memory | IPC (pipes, queues) |
| GIL Impact (Python) | Limited CPU parallelism | True parallelism |
| Fault Isolation | One crash = all crash | Process isolation |
| **Use Cases** | I/O-bound, shared state | CPU-bound, fault tolerance |

### **Parallel Execution Techniques**
- **Pool Executors**: Managed thread/process pools
- **Numba JIT**: Just-in-time compilation for performance
- **Async/Await**: Cooperative concurrency for I/O-bound tasks

---

## **Lecture 8: GPU Programming**

### **GPU Architecture and Purpose**

**GPU vs CPU Philosophy**:
- **CPU**: Optimize for latency (minimize time per task)
- **GPU**: Optimize for throughput (maximize total work done)

**SIMT (Single Instruction, Multiple Threads)**:
- **32 threads per warp** execute same instruction
- **Branch divergence**: Different control flow within warp reduces efficiency
- **Thousands of threads** hide memory latency

### **GPU Memory Hierarchy**
```
Global Memory (GDDR) ←── Slow, large, accessible by all threads
    ↑
L2 Cache ←── Shared across Streaming Multiprocessors
    ↑
Streaming Multiprocessor (SM) ←── ~2048 threads per SM
    ↑
Shared Memory ←── Fast, small, shared within thread block
    ↑
Registers ←── Fastest, private per thread
```

### **GPU Programming Models**

**CUDA Hierarchy**: Grid → Blocks → Threads
```python
# Numba CUDA kernel
from numba import cuda

@cuda.jit
def gpu_kernel(data, result):
    # Global thread index
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < data.size:
        result[idx] = data[idx] ** 2

# Launch: gpu_kernel[blocks, threads_per_block](data, result)
```

**PyTorch GPU Programming**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
result = a @ b  # Matrix multiplication on GPU
```

**PyOpenCL**: Cross-platform GPU programming
- Works with NVIDIA, AMD, Intel GPUs
- Lower-level than CUDA/PyTorch
- Portable across different vendors

### **Performance Optimization**

**Memory Access Patterns**:
- **Coalesced Access**: Threads in warp access contiguous memory
- **Strided Access**: Poor performance due to memory system
- **Random Access**: Worst case for GPU memory

**Shared Memory Usage**:
- Cache frequently accessed data in shared memory
- Coordinate data loading across threads in block
- Much faster than global memory access

**Occupancy**: Balance threads per block vs resource usage
- More threads = better latency hiding
- More resources per thread = fewer concurrent threads
- Find optimal balance for your algorithm

**Branch Divergence Avoidance**:
- Use predication instead of branches when possible
- Group similar work together to reduce divergence
- Profile to identify divergence hotspots

### **When to Use GPU vs CPU**

**GPU Excels**:
✅ Data-parallel problems (same operation on many elements)
✅ Regular memory access patterns
✅ High arithmetic intensity (many ops per memory access)
✅ Large problem sizes (amortize GPU overhead)

**CPU Better**:
❌ Complex control flow (many branches)
❌ Sequential dependencies
❌ Small problem sizes
❌ Irregular memory access patterns

### **Applications and Performance**
- **Machine Learning**: Training neural networks, inference
- **Scientific Computing**: Molecular dynamics, climate modeling
- **Image/Signal Processing**: Filtering, FFTs, computer vision
- **Cryptocurrency**: Mining, blockchain validation
- **Graphics**: Rendering, real-time visualization

---

## **Lecture 9: Testing and Verification**

### **Complexity of Parallel Programs**

**Why Parallel Testing is Hard**:
- **Non-deterministic behavior**: Same input, different execution orders
- **Heisenbugs**: Bugs disappear when observed (adding prints changes timing)
- **State space explosion**: 2ⁿ possible execution orders for n sync points
- **Timing dependencies**: Results depend on hardware, load, OS scheduling

**Sources of Complexity**:
- **Interaction complexity**: Exponential growth with sync points
- **Race conditions**: Outcome depends on thread scheduling
- **Communication errors**: Message loss, ordering issues
- **Load balancing**: Uneven work distribution

### **Modular Design Principles**

**KISS (Keep It Simple, Stupid)**:
- Start with sequential version
- Add parallelism incrementally
- Minimize shared state
- Use high-level abstractions when possible

**Design Patterns**:
- **Producer-Consumer**: Clean separation of data generation/processing
- **Pipeline**: Sequential stages, each handling different data
- **Master-Worker**: Central coordinator distributes tasks

**Reduce Interaction Surface**:
- Fewer shared variables = fewer potential race conditions
- Clear interfaces between components
- Immutable data structures when possible

### **Testing Strategies**

**Passive Testing** (no execution modification):
- **Static Analysis**: Code review, lint tools, type checking
- **Code Review Patterns**: Lock ordering, exception safety
- **Deterministic Replay**: Record execution order for debugging

**Dynamic Testing** (observe during execution):
- **Stress Testing**: Many iterations with random timing
- **Property-Based Testing**: Verify invariants hold under all conditions
- **Concurrency Testing**: Specific tests for thread interactions

### **Types of Testing**

**Unit Testing**: Test individual components
```python
def test_thread_safe_counter():
    counter = ThreadSafeCounter()
    threads = []
    for _ in range(10):
        t = threading.Thread(target=lambda: [counter.increment() for _ in range(1000)])
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    assert counter.value == 10000  # Should be exactly 10,000
```

**Integration Testing**: Test component interactions
- Producer-consumer coordination
- Pipeline stage handoffs  
- Distributed system communication

**Performance Testing**: Measure parallel performance
- **Speedup**: Does parallel version actually run faster?
- **Scalability**: How does performance change with more processors?
- **Efficiency**: Are processors being used effectively?

### **Test-Driven Development (TDD) for Parallel Code**

**TDD Cycle**:
1. **Red**: Write failing test (define expected behavior)
2. **Green**: Write minimal code to pass test
3. **Refactor**: Add parallelism while keeping tests passing

**Benefits for Parallel Code**:
- Ensures correctness before optimization
- Catches regressions when adding parallelism
- Forces thinking about interface design

### **Profilers and Performance Analysis**

**Why Profiling is Critical**:
- Parallel programs often have hidden bottlenecks
- Amdahl's Law: Small sequential sections limit speedup
- Lock contention can make parallel code slower than sequential

**Python Profiling Tools**:
- **cProfile**: Function-level performance analysis
- **line_profiler**: Line-by-line timing analysis  
- **memory_profiler**: Track memory usage over time
- **py-spy**: Sampling profiler for production systems

**What to Look For**:
- **Lock contention**: Time spent waiting for locks
- **Load imbalance**: Some threads finishing much earlier
- **Communication overhead**: Time spent in message passing
- **Memory bandwidth saturation**: Performance plateau despite more cores

**Performance Bottleneck Identification**:
- **Sequential sections**: Amdahl's Law limitations
- **Synchronization overhead**: Too much coordination
- **False sharing**: Cache line bouncing between cores
- **NUMA effects**: Poor memory locality

---

## **Quick Reference Formulas**

**Performance Metrics**:
- Speedup: S_p = T_1 / T_p
- Efficiency: E_p = S_p / p
- Amdahl's Law: S_p = 1 / (f + (1-f)/p)
- Gustafson's Law: S_p = p - f(p-1)

**Communication Cost**: startup cost + m × tw

**Parallel Algorithms**:
- Broadcast/Reduce: O(log p) time
- Parallel Prefix: O(log n) time, O(n) work
- Matrix Multiply: O(n³/p + log p) time

**Critical Path**: Lower bound on parallel execution time
**Granularity Trade-off**: Task size vs overhead balance

**Performance Metrics**:
- Speedup: S_p = T_1 / T_p
- Efficiency: E_p = S_p / p
- Amdahl's Law: S_p = 1 / (f + (1-f)/p)
- Gustafson's Law: S_p = p - f(p-1)

**Communication Cost**: startup cost + m × tw

**Parallel Algorithms**:
- Broadcast/Reduce: O(log p) time
- Parallel Prefix: O(log n) time, O(n) work
- Matrix Multiply: O(n³/p + log p) time

**Critical Path**: Lower bound on parallel execution time
**Granularity Trade-off**: Task size vs overhead balance

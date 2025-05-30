import numpy as np
import time
from numba import prange, jit, vectorize

def fill(N,K,M,n_exp):
    x = np.random.randn(1, N)     # Filling vector
    A = np.random.randn(N, N)     # Filling matrix
    if n_exp ==1: 
        print(f'Vector\n {x}')
        print(f'Matrix\n {A}')
    return x, A
    
def timer(f):                                   # Decorator to calculate execution time of a funciton
    def wrapper(*args, **kw):                   # Needed to decorate a function with input arguments
        t_start = time.time()
        result = f(*args, **kw)                 # Calling function
        t_end = time.time()
        return result, t_end-t_start            # Return the result AND the execution time
    return wrapper


@timer       
def numpy_mult(x,A):
    y = np.matmul(x,A)
    return y
    
@timer
@jit(nopython=True, parallel=False)    
def numba_jit_mult(x,A):        
    H, K  = A.shape 
    y = np.zeros(K, dtype=np.float64)
    for n in range(H):
        for k in range(K):
            y[k] += x[0,n]*A[n,k]
    return y


@vectorize(['float64(float64, float64)'], target='cpu')
def vectorized_hadamard(a,b):
    return a*b


@vectorize(['float64(float64, float64)'], target='cpu')
def vectorized_sum(a,b):
    return a+b    
    
if __name__=="__main__":   
    np.random.seed(2)   # Setting random seed
    N = 100             # Size of vector (Reduceret til 100 fordi fuck nej 10000)
    K = N               # No. columns in matrix
    n_vals = 10         # The range of values is [0,n_vals-1]
    n_exp = 5           # Number of realizations (Reduceret til 5 fordi fuck nej 50)
    exec_times = np.zeros(3)
    max_val = 10        # Upper limit for random number generation is max_val-1
    for i in range(n_exp):       
        x, A = fill(N,K, max_val, n_exp)

        # Calculate the product with numpy

        y_numpy, t_numpy = numpy_mult(x,A)
        y_numpy = y_numpy.squeeze() #shape to (N,)

        # Calculate the product with JIT
        
        y_jit, t_jit = numba_jit_mult(x,A) 
        
        
        # Calculating the product with vectorized decorator
        xf = x.copy().squeeze() # x is (1, N) and A is (N, N) no work so squeeze x to (N,)
        Af = A.copy() #Dont cast to float32 
        xf = xf.reshape(N,1) #now reshape x into (N, 1) now when we multiply we get a (N, N) result
        t_start = time.time()
        Y_vector = vectorized_hadamard(xf,Af)
        y_vector = vectorized_sum.reduce(Y_vector, axis=0)
        t_end = time.time()
        t_vector= t_end-t_start
        
        exec_times += [t_numpy, t_jit, t_vector]
    exec_times/=n_exp

    print(f'\n%%%%\nFinished execution\n')
    if np.allclose(y_numpy,y_jit) and np.allclose(y_numpy,y_vector):   # Uncomment to execute
        print(f'All the results are correct\nAverage elapsed time in milliseconds:')
        print('    Numpy: {:0.3e} '.format(exec_times[0]*1e3))
        print('    JIT: {:0.3e} '.format(exec_times[1]*1e3))
        print('    Vectorization: {:0.3e} '.format(exec_times[2]*1e3))
    else:
        print(f'Results are incorrect!!!')     

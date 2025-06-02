# Import system utilities for file operations and command line arguments
import sys, os, time, argparse
# Import NumPy for numerical array operations
import numpy as np
# Import Numba decorators for JIT compilation and vectorization
from numba import float32, guvectorize, njit, prange
# Import multiprocessing tools for parallel execution across CPU cores
from multiprocessing import Pool, cpu_count
# Import PIL for image loading, saving, and format conversion
from PIL import Image

# 1. Precompute DCT transform matrix

def compute_dct_matrix(K: int) -> np.ndarray:
    """
    Compute the K x K DCT (type II) matrix T.
    T[k, n] = alpha_k * cos(pi*k*(2*n+1)/(2*K))
    """
    # Initialize K×K matrix with zeros, using float32 for memory efficiency
    T = np.zeros((K, K), dtype=np.float32)
    # Precompute the constant factor π/(2K) used in DCT formula
    factor = np.pi / (2 * K)
    # Loop through each row k of the DCT matrix
    for k in range(K):
        # DC component (k=0) has different normalization than AC components
        alpha = np.sqrt(1.0 / K) if k == 0 else np.sqrt(2.0 / K)
        # Loop through each column n of the DCT matrix
        for n in range(K):
            # Compute DCT-II basis function: alpha * cos(π*k*(2n+1)/(2K))
            T[k, n] = alpha * np.cos((2*n + 1) * k * factor)
    # Return the completed DCT transformation matrix
    return T

# Standard JPEG 8x8 luminance quantization matrix - smaller values preserve more detail
Q8 = np.array([
    # Low frequencies (top-left) have small values - preserve important visual info
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    # High frequencies (bottom-right) have large values - heavily quantize less visible details
    [72, 92, 95, 98,112, 100, 103,  99]
], dtype=np.float32)  # Use float32 to match other matrix operations

# --- Sequential NumPy implementation ---

def dct2_block_numpy(block: np.ndarray, T: np.ndarray) -> np.ndarray:
    # Perform 2D DCT using matrix multiplication: T * block * T^T
    return T @ block @ T.T

def idct2_block_numpy(block: np.ndarray, T: np.ndarray) -> np.ndarray:
    # Perform 2D inverse DCT using matrix multiplication: T^T * block * T
    return T.T @ block @ T

def jpeg_blockwise_numpy(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    # Convert image to float32 and center pixel values around 0 (JPEG standard)
    X = image.astype(np.float32) - 128.0
    # Get image dimensions for block iteration
    H, W = X.shape
    # Compute DCT transformation matrix for the specified block height
    T_h = compute_dct_matrix(block_h)
    # Initialize output array with same shape as input, filled with zeros
    out = np.zeros_like(X)
    # Iterate through image in non-overlapping blocks
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            # Extract current block from image
            block = X[i:i+block_h, j:j+block_w]
            # Apply forward DCT to convert spatial domain to frequency domain
            Y = dct2_block_numpy(block, T_h)
            # Quantize DCT coefficients by dividing by quantization matrix and rounding
            Yq = np.round(Y / Q)
            # Dequantize by multiplying back (simulates decompression step)
            Ydq = Yq * Q
            # Apply inverse DCT to convert back to spatial domain
            rec = idct2_block_numpy(Ydq, T_h)
            # Place reconstructed block back into output image
            out[i:i+block_h, j:j+block_w] = rec
    # Add back 128 offset, clamp to valid pixel range [0,255], convert to uint8
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# --- Sequential Numba Vectorization implementation ---

# Numba guvectorize decorator compiles this to optimized machine code
# Signature: takes two 2D float32 arrays as input, produces one 2D float32 array as output
@guvectorize([(float32[:,:], float32[:,:], float32[:,:])], '(n,n),(n,n)->(n,n)')
def dct2_block_vector(block, T, Y):
    # Get block dimensions (assumed square)
    K = block.shape[0]
    # Nested loops to compute 2D DCT manually (equivalent to T @ block @ T.T)
    for k in range(K):          # Output row index
        for l in range(K):      # Output column index
            # Initialize accumulator for this output element
            Y[k, l] = 0.0
            # Inner loops perform the matrix multiplication
            for n in range(K):  # Sum over block rows
                for m in range(K):  # Sum over block columns
                    # Accumulate: T[k,n] * block[n,m] * T[l,m]
                    Y[k, l] += T[k, n] * block[n, m] * T[l, m]

# Inverse DCT using same vectorization approach
@guvectorize([(float32[:,:], float32[:,:], float32[:,:])], '(n,n),(n,n)->(n,n)')
def idct2_block_vector(block, T, Y):
    # Get block dimensions
    K = block.shape[0]
    # Nested loops to compute 2D inverse DCT (equivalent to T.T @ block @ T)
    for k in range(K):          # Output row index
        for l in range(K):      # Output column index
            # Initialize accumulator
            Y[k, l] = 0.0
            # Inner loops for inverse transformation
            for n in range(K):
                for m in range(K):
                    # Accumulate: T[n,k] * block[n,m] * T[m,l] (note transposed T indices)
                    Y[k, l] += T[n, k] * block[n, m] * T[m, l]

def jpeg_blockwise_vector(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    # Center pixel values around 0
    X = image.astype(np.float32) - 128.0
    # Get image dimensions
    H, W = X.shape
    # Compute DCT matrix once
    T_h = compute_dct_matrix(block_h)
    # Initialize output array
    out = np.zeros_like(X)
    # Process each block
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            # Extract current block
            block = X[i:i+block_h, j:j+block_w]
            # Create temporary array for DCT coefficients
            Y = np.zeros((block_h, block_w), dtype=np.float32)
            # Apply forward DCT using vectorized function
            dct2_block_vector(block, T_h, Y)
            # Quantize coefficients (rint = round to nearest integer)
            Yq = np.rint(Y / Q)
            # decompress - dequantize coefficients
            Ydq = Yq * Q
            # Apply inverse DCT, reusing Y array for output
            rec = idct2_block_vector(Ydq, T_h, Y)
            # Store reconstructed block in output image
            out[i:i+block_h, j:j+block_w] = rec
    # Convert back to uint8 pixel values
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# --- Sequential Numba JIT implementation ---

# Numba njit decorator provides just-in-time compilation for speed
@njit
def dct2_block_numba(block, T, Y):
    # Get block size
    K = block.shape[0]
    # Compute 2D DCT using explicit loops
    for k in range(K):
        for l in range(K):
            # Use local variable for accumulation (may be faster)
            s = 0.0
            # Inner loops for matrix multiplication
            for n in range(K):
                for m in range(K):
                    # Accumulate contribution to output element [k,l]
                    s += T[k, n] * block[n, m] * T[l, m]
            # Store final result
            Y[k, l] = s

@njit
def idct2_block_numba(block, T, Y):
    # Get block size
    K = block.shape[0]
    # Compute 2D inverse DCT
    for k in range(K):
        for l in range(K):
            # Local accumulator
            s = 0.0
            # Inner loops for inverse transformation
            for n in range(K):
                for m in range(K):
                    # Note: T indices are transposed for inverse DCT
                    s += T[n, k] * block[n, m] * T[m, l]
            # Store result
            Y[k, l] = s

def jpeg_blockwise_numba(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    # Center pixels around 0
    X = image.astype(np.float32) - 128.0
    # Get dimensions
    H, W = X.shape
    # Compute DCT matrix
    T_h = compute_dct_matrix(block_h)
    # Initialize output
    out = np.zeros_like(X)
    # Process blocks sequentially
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            # Extract block
            block = X[i:i+block_h, j:j+block_w]
            # Create temporary for DCT coefficients
            Y = np.zeros((block_h, block_w), dtype=np.float32)
            # Forward DCT
            dct2_block_numba(block, T_h, Y)
            # Quantize (rint rounds to nearest integer)
            Yq = np.rint(Y / Q)
            # decompress - dequantize
            Ydq = Yq * Q
            # Inverse DCT (overwrites Y with spatial domain result)
            idct2_block_numba(Ydq, T_h, Y)
            # Copy result to output (Y now contains spatial domain data)
            out[i:i+block_h, j:j+block_w] = Y
    # Convert to uint8 pixels
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# --- Parallel Numba (parallel=True) implementation ---

# parallel=True enables automatic parallelization across CPU cores
@njit(parallel=True)
def jpeg_blockwise_numba_parallel(X, T_h, Q, out, block_h, block_w):
    # Get image dimensions
    H, W = X.shape
    # Calculate number of blocks in each dimension
    nbi = H // block_h  # Number of block rows
    nbj = W // block_w  # Number of block columns
    # prange enables parallel execution of outer loop across CPU threads
    for bi_idx in prange(nbi):
        # Inner loop runs sequentially within each thread
        for bj_idx in range(nbj):
            # Convert block indices to pixel coordinates
            bi = bi_idx * block_h
            bj = bj_idx * block_w
            # Extract block from input image
            block = X[bi:bi+block_h, bj:bj+block_w]
            # Create temporary array for this thread
            Y = np.zeros((block_h, block_w), dtype=np.float32)
            # Forward DCT
            dct2_block_numba(block, T_h, Y)
            # Quantize
            Yq = np.rint(Y / Q)
            # Dequantize
            Ydq = Yq * Q
            # Inverse DCT
            idct2_block_numba(Ydq, T_h, Y)
            # Write result to output array (thread-safe since blocks don't overlap)
            out[bi:bi+block_h, bj:bj+block_w] = Y

def jpeg_blockwise_numba_par(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    # Center pixels
    X = image.astype(np.float32) - 128.0
    # Get dimensions
    H, W = X.shape
    # Compute DCT matrix
    T_h = compute_dct_matrix(block_h)
    # Initialize output
    out = np.zeros_like(X)
    # Call parallel processing function
    jpeg_blockwise_numba_parallel(X, T_h, Q, out, block_h, block_w)
    # Convert result to uint8
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# --- Multiprocessing implementation ---

def _process_one_block(args):
    # Unpack arguments passed from main process
    i, j, X, block_h, block_w, T_h, Q = args
    # Extract the specific block to process
    block = X[i:i+block_h, j:j+block_w]
    # Forward DCT using matrix multiplication (NumPy)
    Y = T_h @ block @ T_h.T
    # Quantize coefficients
    Yq = np.round(Y / Q)
    # Dequantize
    Ydq = Yq * Q
    # Inverse DCT
    rec = T_h.T @ Ydq @ T_h
    # Return block coordinates and reconstructed data
    return (i, j, rec)

def jpeg_blockwise_mp(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    # Center pixels
    X = image.astype(np.float32) - 128.0
    # Get dimensions
    H, W = X.shape
    # Compute DCT matrix
    T_h = compute_dct_matrix(block_h)
    # Initialize output
    out = np.zeros_like(X)
    # Create list of tasks - each task contains ALL data needed for one block
    # WARNING: This is inefficient - sends entire image X with every task!
    tasks = [(i, j, X, block_h, block_w, T_h, Q)
             for i in range(0, H, block_h)     # Block row positions
             for j in range(0, W, block_w)]    # Block column positions
    # Create process pool with all available CPU cores
    with Pool(cpu_count()) as pool:
        # Map tasks to worker processes and collect results
        for i, j, rec in pool.map(_process_one_block, tasks):
            # Place reconstructed block into output image
            out[i:i+block_h, j:j+block_w] = rec
    # Convert to uint8
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser()
    # Required positional argument: path to input image file
    parser.add_argument("input_image")
    # Required method selection with predefined choices
    parser.add_argument("--method",
        choices=["vector","numba","numba-parallel","mp"],
        required=True)
    # Optional: create NxN grid of input image for larger test images
    parser.add_argument("--grid", type=int, default=1)
    # Optional: number of times to repeat processing for timing accuracy
    parser.add_argument("--repeat", type=int, default=1)
    # Parse command line arguments
    args = parser.parse_args()

    # load grayscale input - open image file and convert to luminance only
    inp = Image.open(args.input_image).convert("L")
    # tile into a grid for performance testing with larger images
    W, H = inp.size  # Get original image dimensions
    # Create new image canvas that's grid×grid times larger
    img = Image.new("L", (W*args.grid, H*args.grid))
    # Fill the canvas by tiling the original image
    for i in range(args.grid):
        for j in range(args.grid):
            # Paste original image at position (i*W, j*H)
            img.paste(inp, (i*W, j*H))
    
    # convert to array for processing - PIL Image to NumPy array
    arr = np.array(img, dtype=np.uint8)
    # Get final image dimensions after tiling
    H, W = arr.shape
    # Create resolution string for output filename
    res = f"{H}x{W}"
    
    # Dictionary mapping method names to actual functions
    mapping = {
      "vector": jpeg_blockwise_vector,
      "numba": jpeg_blockwise_numba,
      "numba-parallel": jpeg_blockwise_numba_par,
      "mp": jpeg_blockwise_mp,
    }
    # Select the compression function based on command line argument
    func = mapping[args.method]

    # Performance timing section
    t0 = time.perf_counter()    # Start high-precision timer
    # Run compression multiple times for better timing accuracy
    for r in range(args.repeat):
        # Call selected compression function with 8x8 blocks and standard quantization
        rec = func(arr, 8, 8, Q8)  # rec = reconstructed image
    # Stop timer and calculate elapsed time
    dt = time.perf_counter()-t0
    # Print results in CSV format: method,resolution,time
    print(f"{args.method},{res},{dt:.4f}")

    # Save output image
    os.makedirs("output", exist_ok=True)  # Create output directory if it doesn't exist
    # Generate output filename with method, resolution, and repeat count
    out_fname = os.path.join("output", f"{args.method}_{res}_{args.repeat}.png")
    # Convert NumPy array back to PIL Image
    out_img = Image.fromarray(rec)
    # Save compressed result as PNG file
    out_img.save(out_fname)
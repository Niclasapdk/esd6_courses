import sys, os, time, argparse
import numpy as np
from numba import float32, guvectorize, njit, prange
from multiprocessing import Pool, cpu_count
from PIL import Image

# 1. Precompute DCT transform matrix

def compute_dct_matrix(K: int) -> np.ndarray:
    """
    Compute the K x K DCT (type II) matrix T.
    T[k, n] = alpha_k * cos(pi*k*(2*n+1)/(2*K))
    """
    T = np.zeros((K, K), dtype=np.float32)
    factor = np.pi / (2 * K)
    for k in range(K):
        alpha = np.sqrt(1.0 / K) if k == 0 else np.sqrt(2.0 / K)
        for n in range(K):
            T[k, n] = alpha * np.cos((2*n + 1) * k * factor)
    return T

# Standard JPEG 8x8 luminance quantization matrix
Q8 = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
], dtype=np.float32)

# --- Sequential NumPy implementation ---

def dct2_block_numpy(block: np.ndarray, T: np.ndarray) -> np.ndarray:
    return T @ block @ T.T

def idct2_block_numpy(block: np.ndarray, T: np.ndarray) -> np.ndarray:
    return T.T @ block @ T

def jpeg_blockwise_numpy(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    X = image.astype(np.float32) - 128.0
    H, W = X.shape
    T_h = compute_dct_matrix(block_h)
    out = np.zeros_like(X)
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            block = X[i:i+block_h, j:j+block_w]
            Y = dct2_block_numpy(block, T_h)
            Yq = np.round(Y / Q)
            Ydq = Yq * Q
            rec = idct2_block_numpy(Ydq, T_h)
            out[i:i+block_h, j:j+block_w] = rec
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)
# --- Sequential Numba Vectorization implementation ---

@guvectorize([(float32[:,:], float32[:,:], float32[:,:])], '(n,n),(n,n)->(n,n)')
def dct2_block_vector(block, T, Y):
    K = block.shape[0]
    for k in range(K):
        for l in range(K):
            Y[k, l] = 0.0
            for n in range(K):
                for m in range(K):
                    Y[k, l] += T[k, n] * block[n, m] * T[l, m]

@guvectorize([(float32[:,:], float32[:,:], float32[:,:])], '(n,n),(n,n)->(n,n)')
def idct2_block_vector(block, T, Y):
    K = block.shape[0]
    for k in range(K):
        for l in range(K):
            Y[k, l] = 0.0
            for n in range(K):
                for m in range(K):
                    Y[k, l] += T[n, k] * block[n, m] * T[m, l]

def jpeg_blockwise_vector(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    X = image.astype(np.float32) - 128.0
    H, W = X.shape
    T_h = compute_dct_matrix(block_h)
    out = np.zeros_like(X)
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            block = X[i:i+block_h, j:j+block_w]
            Y = np.zeros((block_h, block_w), dtype=np.float32)
            dct2_block_vector(block, T_h, Y)
            Yq = np.rint(Y / Q)
            # decompress
            Ydq = Yq * Q
            rec = idct2_block_vector(Ydq, T_h, Y)
            out[i:i+block_h, j:j+block_w] = rec
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# --- Sequential Numba JIT implementation ---

@njit
def dct2_block_numba(block, T, Y):
    K = block.shape[0]
    for k in range(K):
        for l in range(K):
            s = 0.0
            for n in range(K):
                for m in range(K):
                    s += T[k, n] * block[n, m] * T[l, m]
            Y[k, l] = s

@njit
def idct2_block_numba(block, T, Y):
    K = block.shape[0]
    for k in range(K):
        for l in range(K):
            s = 0.0
            for n in range(K):
                for m in range(K):
                    s += T[n, k] * block[n, m] * T[m, l]
            Y[k, l] = s

def jpeg_blockwise_numba(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    X = image.astype(np.float32) - 128.0
    H, W = X.shape
    T_h = compute_dct_matrix(block_h)
    out = np.zeros_like(X)
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            block = X[i:i+block_h, j:j+block_w]
            Y = np.zeros((block_h, block_w), dtype=np.float32)
            dct2_block_numba(block, T_h, Y)
            Yq = np.rint(Y / Q)
            # decompress
            Ydq = Yq * Q
            idct2_block_numba(Ydq, T_h, Y)
            out[i:i+block_h, j:j+block_w] = Y
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# --- Parallel Numba (parallel=True) implementation ---

@njit(parallel=True)
def jpeg_blockwise_numba_parallel(X, T_h, Q, out, block_h, block_w):
    H, W = X.shape
    nbi = H // block_h
    nbj = W // block_w
    for bi_idx in prange(nbi):
        for bj_idx in range(nbj):
            bi = bi_idx * block_h
            bj = bj_idx * block_w
            block = X[bi:bi+block_h, bj:bj+block_w]
            Y = np.zeros((block_h, block_w), dtype=np.float32)
            dct2_block_numba(block, T_h, Y)
            Yq = np.rint(Y / Q)
            Ydq = Yq * Q
            idct2_block_numba(Ydq, T_h, Y)
            out[bi:bi+block_h, bj:bj+block_w] = Y


def jpeg_blockwise_numba_par(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    X = image.astype(np.float32) - 128.0
    H, W = X.shape
    T_h = compute_dct_matrix(block_h)
    out = np.zeros_like(X)
    jpeg_blockwise_numba_parallel(X, T_h, Q, out, block_h, block_w)
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# --- Multiprocessing implementation ---

def _process_one_block(args):
    i, j, X, block_h, block_w, T_h, Q = args
    block = X[i:i+block_h, j:j+block_w]
    Y = T_h @ block @ T_h.T
    Yq = np.round(Y / Q)
    Ydq = Yq * Q
    rec = T_h.T @ Ydq @ T_h
    return (i, j, rec)

def jpeg_blockwise_mp(image: np.ndarray, block_h: int, block_w: int, Q: np.ndarray) -> np.ndarray:
    X = image.astype(np.float32) - 128.0
    H, W = X.shape
    T_h = compute_dct_matrix(block_h)
    out = np.zeros_like(X)
    tasks = [(i, j, X, block_h, block_w, T_h, Q)
             for i in range(0, H, block_h)
             for j in range(0, W, block_w)]
    with Pool(cpu_count()) as pool:
        for i, j, rec in pool.map(_process_one_block, tasks):
            out[i:i+block_h, j:j+block_w] = rec
    return np.clip(out + 128.0, 0, 255).astype(np.uint8)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image")
    parser.add_argument("--method",
        choices=["vector","numba","numba-parallel","mp"],
        required=True)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    # load grayscale input
    inp = Image.open(args.input_image).convert("L")
    # tile into a 4x4 grid
    W, H = inp.size
    img = Image.new("L", (W*args.grid, H*args.grid))
    for i in range(args.grid):
        for j in range(args.grid):
            img.paste(inp, (i*W, j*H))
    # convert to array for processing
    arr = np.array(img, dtype=np.uint8)
    H, W = arr.shape
    res = f"{H}x{W}"
    mapping = {
      "vector": jpeg_blockwise_vector,
      "numba": jpeg_blockwise_numba,
      "numba-parallel": jpeg_blockwise_numba_par,
      "mp": jpeg_blockwise_mp,
    }
    func = mapping[args.method]

    t0 = time.perf_counter()
    for r in range(args.repeat):
        rec = func(arr, 8, 8, Q8)  # or Q_exp, depending on your code
    dt = time.perf_counter()-t0
    print(f"{args.method},{res},{dt:.4f}")

    os.makedirs("output", exist_ok=True)
    out_fname = os.path.join("output", f"{args.method}_{res}_{args.repeat}.png")
    out_img = Image.fromarray(rec)
    out_img.save(out_fname)

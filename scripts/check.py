# checks the gpu attention output against a float64 NumPy reference.
# usage: python3 scripts/check.py data/Q.bin data/K.bin data/V.bin result.bin
import sys
import numpy as np
    
S, D = 128, 768

def load(path):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != S * D:
        sys.exit(f"{path}: {a.size} floats, expected {S * D}")
    return a.reshape(S, D).astype(np.float64)

q, k, v, out = map(load, sys.argv[1:5])

s = q @ k.T / np.sqrt(D)
s -= s.max(axis=1, keepdims=True)
p = np.exp(s)
p /= p.sum(axis=1, keepdims=True)
ref = p @ v

diff = out - ref
print(f"max abs error : {np.abs(diff).max():.3e}")
print(f"rel L2 error  : {np.linalg.norm(diff) / np.linalg.norm(ref):.3e}")
print("PASS" if np.allclose(out, ref, rtol=1e-4, atol=1e-5) else "FAIL")

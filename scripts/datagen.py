import numpy as np

# If you use these, you need to update the constants in main.cc
seq_len = 1024
head_dim = 8192

np.random.seed(42)

Q = np.random.randn(seq_len, head_dim).astype(np.float32)
K = np.random.randn(seq_len, head_dim).astype(np.float32)
V = np.random.randn(seq_len, head_dim).astype(np.float32)

Q.tofile("../data/Q.bin")
K.tofile("../data/K.bin")
V.tofile("../data/V.bin")

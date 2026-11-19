# Assignment CUDA: Scaled Dot-Product Attention
In this directory, you can find the framework for the assignment required to implement Scaled Dot-Product Attention which is the foundational mechanism behind the Transformer architecture.

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

You will be implementing the operations required to calculate attention using CUDA kernels: two matrix multiplications, normalization and (safe) softmax.

## Framework information
The ``src`` directory contains the source code you will be working with, specifically attention.cu. This file is where you will be implementing your CUDA program for calculating attention. Please do not change any of the other files outside from temporary benchmarking/experimenting purposes (e.g. debug functions and such in ``main.cc`` for yourself). Your ``main.cc`` will be replaced during grading, so refrain from implementing required functionality in there. There is a sample CUDA program implemented in ``vector-add.cu``, you can use the code in this file for inspiration.

If you want to add extra files that need extra compilation, that is allowed, but please change the Makefile and hand it in, and be sure to note in your report that you did. Additionally, by default, the framework runs on a GTX Titan X on the DAS5, which has compute capability 5.2.
If you want to use another GPU, or want to run locally on another GPU, please change the COMPUTE variable in the Makefile to match the correct compute
capability (this should be the version minus the dot - so for the Titan X, this
becomes '52'). Alternatively, if you append 'COMPUTE=X' to your make command,
then the compute capability is set to value X for one run only. Note that you
may have to rebuild from scratch for it to work, i.e., run 'make clean' first.
CUDA compute capabilities for various GPUs can be found on this page [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)

### Models
The models to be used can be locally stored under the models directory. Currently, this assignment is designed to work with [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/blob/main/model.onnx). However, you can add other models you want to experiment with, as long as the code submitted supports this model.

### Scripts
To extract and prepare the data required for running the CUDA program a Python script is provided in the scripts directory. The script makes use of the numpy, onnx and onnxruntime Python libraries. This script extracts the Q, K, V from the first attention layer. These are stored in both ``.npy`` and ``.bin`` files for ease of use. There is also a generation script for random tensors that can be used for benchmarking. You should run the scripts locally and only copy over the data files to the DAS5.

The scripts have a few dependencies, so you should install them if you don't have them yet:
```
pip install numpy onnx onnxruntime
```

### Repository overview
```
data/              # Binary and NumPy tensors for experimenting extracted from the models
models/            # ONNX model files
README.md          # Project documentation
scripts/           # Data extraction script
src/               # C++ and CUDA source code
```

## Compiling & Running

### Required modules on DAS5
Use the following commands to load the appropriate modules:
```
module load prun
```
```
module load cuda12.6/toolkit
```

### Compiling
To compile the framework (add 'COMPUTE=NUM' behind these commands if you are targetting another compute capability, see the Makefile):
```
make
```
or
```
make attention
```

If you want to build the vector-add example you can use:
```
make vector-add
```

### Running
To run the framework locally with a few test parameters (check 'ARGS' in the Makefile):
```
make runlocal
```
or to run it on the DAS5
```
make run
```

### Utilities
If you encounter weird behaviour or errors, try building from scratch by removing all compiled files and results by running:
```
make clean
```

To create your submission tarball you can use the following command (do double-check afterwards whether everything is included):
```
make tarball
```

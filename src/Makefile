# Compilers
NVCC = nvcc
CC   = g++

# Change this to change the GPU compute capability the compiler will compile against
# GTX TitanX uses 52
COMPUTE ?= 52

# Default arguments
ARGS ?= -b 1000 -v

# Compiler flags
CC_FLAGS = -O3 -m64 -Wall -std=c++20
CU_FLAGS = -O3 -g --ptxas-options=-v -arch compute_$(COMPUTE) -code sm_$(COMPUTE) -std=c++20
# CU_FLAGS = -O3 -g --ptxas-options=-v -arch=sm_$(COMPUTE) -std=c++20

# Define the source files
CU_SOURCES = attention.cu
CC_SOURCES = main.cc timer.cc

# Define tarball related
TARBALL = attention.tar.gz
HEADER_FILES = attention.hh timer.hh
FILES = $(CU_SOURCES) $(CC_SOURCES) $(HEADER_FILES) Makefile

# Create paths to their relevant object files
CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

# Targets
all: attention

attention: $(CU_OBJECTS) $(CC_OBJECTS)
	$(NVCC) $^ -o $@

vector-add: vector-add.o timer.o
	$(NVCC) $^ -o $@

# Compile objects; either as CUDA objects (.cu), C++ objects (.cc) or GPU assembly
%.o: %.cu
	$(NVCC) $(CU_FLAGS) -c $< -o $@
%.o: %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@
%.ptx: %.cu
	$(NVCC) $(CU_FLAGS) --ptx $< -o $@

# Compile to GPU-instructions only
ptx: $(CU_PTX)

clean:
	rm -f *.o *.ptx attention result.bin attention.tar.gz vector-add

runlocal:
	@make all
	@echo ""
	./attention $(ARGS)

run:
	prun -v -np 1 -native "-C TitanX --gres=gres:gpu:1" ./attention $(ARGS)

tarball:
	tar -czvf $(TARBALL) $(FILES)

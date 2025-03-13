NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_86

all: vectorAdd

vectorAdd: vectorAdd.cu
	$(NVCC) $(NVCC_FLAGS) -o vectorAdd vectorAdd.cu

clean:
	rm -f vectorAdd
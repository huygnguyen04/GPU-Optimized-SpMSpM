
NVCC        = nvcc
NVCC_FLAGS  = -O3
CXX         = g++
CXX_FLAGS   = -O3
OBJ         = main.o matrix.o kernel0.o kernel1.o kernel2.o kernel3.o kernel4.o
EXE         = spmspm
GEN_COO     = data/gen_coo


default: $(EXE) $(GEN_COO)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

$(GEN_COO): data/gen_coo.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $<

clean:
	rm -rf $(OBJ) $(EXE) $(GEN_COO)
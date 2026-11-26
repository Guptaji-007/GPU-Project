NVCC = nvcc
FLAGS = -O3 -arch=sm_80
INC = -Isrc -IcsrToEll

SRC = kernels/main.cu src/matrix.cpp csrToEll/cpu_ref.cpp
HDR = src/matrix.h csrToEll/cpu_ref.h csrToEll/utils.h

all: bin/bench

bin/bench: $(SRC) $(HDR)
	mkdir -p bin
	$(NVCC) $(FLAGS) $(INC) -o bin/bench $(SRC) -lcusparse

clean:
	rm -rf bin

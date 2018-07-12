CXX = $(shell which clang++)
LINK = $(shell which clang++)
# NVCC = /usr/local/cuda/bin/nvcc
NVCC = $(shell which nvcc)
SRC ?= src
CUDAHOME += /usr/local/cuda-8.0
TERRAHOME ?= ../../../terra
TERRA ?= ../../../terra/release/terra

FLAGS += -g -std=c++11 -Isrc -Isrc/cutil/inc -I../../API/release/include -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I../external/mLib/include -I../external
LFLAGS += -std=c++11 -L../../API/release/lib -lOpt -L$(CUDAHOME)/lib64 -lcudart -L$(TERRAHOME)/lib -lterra -ldl
LFLAGS += -Wl,-rpath,$(CUDAHOME)/lib64 -pthread

all: image_warping

main.o: src/main.cpp
	$(CXX) -c -o main.o src/main.cpp $(FLAGS) -MP -MD

mlib.o: src/mLibSource.cpp
	$(CXX) -c -o mlib.o src/mLibSource.cpp $(FLAGS) -MP -MD

image_warping: main.o mlib.o
	$(CXX) -o image_warping main.o mlib.o $(LFLAGS) 

clean:
	rm -f image_warping main.o mlib.o
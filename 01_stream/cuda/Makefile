CC=nvcc
CFLAGS=

CUDA_HELPERS=$(wildcard ../../helpers/cuda/*.cu)

build:
	$(CC) $(CFLAGS) -o linear_regression.o linear_regression.cu $(CUDA_HELPERS)

run: build
	./linear_regression.o ../data.csv 100

clean:
	rm *.o

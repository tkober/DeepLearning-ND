__global__ void reduce_add_kernel(const float *d_in, float *d_out, int input_size) {
  extern __shared__ float sdata[];
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int myId = threadIdx.x;

  // Put whole block in shared memory
  sdata[myId] = (index < input_size) ? d_in[index] : 0;
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i>>=1) {
    if (myId < i) {
      sdata[myId] = sdata[myId] + sdata[myId+i];
    }
    __syncthreads();
  }

  if (myId == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}


int next_power_of_two(int number) {
  int result = 1;
  while(result < number) {
    result <<= 1;
  }
  return result;
}


void primitive_reduce_add(float *d_input, float *d_result, int input_size) {
  const int SIZE_AS_POT = next_power_of_two(input_size);

  int shared_size = sizeof(float) * SIZE_AS_POT;
  reduce_add_kernel<<<1, SIZE_AS_POT, shared_size>>>(d_input, d_result, input_size);
}

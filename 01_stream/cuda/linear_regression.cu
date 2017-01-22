#include "../../helpers/csv.h"
#include "../../helpers/cuda/reduce.h"

typedef struct {
  float *x;
  float *y;
  int count;
} Points;


Points load_data(char *file_name, int count) {
  std::ifstream file(file_name);

  Points result = {
    (float *) malloc(sizeof(float) * count),
    (float *) malloc(sizeof(float) * count),
    count
  };

  CSVRow row;
  for (int i = 0; i < result.count; i++) {
    file >> row;
    result.x[i] = atof(row[0].c_str());
    result.y[i] = atof(row[1].c_str());
  }

  return result;
}


__global__ void calculate_error_kernel(
  const float b,
  const float m,
  float const *d_x,
  float const *d_y,
  float *d_output,
  const int size
) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    // (y - (m*x + b)) ** 2
    float error = d_y[index] - (m * d_x[index] + b);
    d_output[index] = pow(error, 2);
  }
}


float calculate_error(
  const float b,
  const float m,
  float const *d_x,
  float const *d_y,
  const int size
) {
  const int MAX_THREADS = 1024;
  const int THREAD_COUNT = min(MAX_THREADS, size);
  const int BLOCK_COUNT = ceil((double) size / (double) THREAD_COUNT);

  float *d_errors;
  cudaMalloc((void **) &d_errors, sizeof(float) * size);

  // Calclulate the errors for all (x, y) pairs
  calculate_error_kernel<<<BLOCK_COUNT, THREAD_COUNT>>>(b, m, d_x, d_y, d_errors, size);

  // Calculate the total error
  float *d_error_sum;
  cudaMalloc((void **) &d_error_sum, sizeof(float));
  primitive_reduce_add(d_errors, d_error_sum, size);

  // Copy the error sum to the host
  float h_error_sum[1];
  cudaMemcpy(h_error_sum, d_error_sum, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_error_sum);
  cudaFree(d_errors);

  // Return mean error
  return h_error_sum[0] / (float) size;
}


__global__ void gradient_step_kernel(
  float const *d_b,
  float const *d_m,
  float const *d_x,
  float const *d_y,
  float *d_b_gradients,
  float *d_m_gradients,
  const int size
) {

}


void gradient_step(
  float *d_b,
  float *d_m,
  float const *d_x,
  float const *d_y,
  const int size,
  const float learning_rate
) {
  const int MAX_THREADS = 1024;
  const int THREAD_COUNT = min(MAX_THREADS, size);
  const int BLOCK_COUNT = ceil((double) size / (double) THREAD_COUNT);

  // Allocate memory on the GPU for the new gradients for each (x, y) pair
  float *d_b_gradients, *d_m_gradients;
  cudaMalloc((void **) &d_b_gradients, sizeof(float) * size);
  cudaMalloc((void **) &d_m_gradients, sizeof(float) * size);

  gradient_step_kernel<<<BLOCK_COUNT, THREAD_COUNT>>>(d_b, d_m, d_x, d_y,
    d_b_gradients, d_m_gradients, size);
}


void gradient_descent(
  float *h_b,
  float *h_m,
  float const *d_x,
  float const *d_y,
  const int size,
  const int number_of_iterations,
  const float learning_rate
) {
  // Copy b and m to GPU
  float *d_b, *d_m;
  cudaMalloc((void**) &d_b, sizeof(float));
  cudaMemcpy(d_b, h_b, sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_m, sizeof(float));
  cudaMemcpy(d_m, h_m, sizeof(float), cudaMemcpyHostToDevice);

  // Perform the gradient steps
  for (int i = 0; i < number_of_iterations; i++) {
    gradient_step(d_b, d_m, d_x, d_y, size, learning_rate);
  }

  // Copy the new values for b and m back to the host
  cudaMemcpy(h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_m, d_m, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_b);
  cudaFree(d_m);
}


int main(int argc, char **argv) {
  // Load the data
  Points points = load_data(argv[1], atoi(argv[2]));

  // Copy the data to the GPU
  float *d_x, *d_y;
  cudaMalloc((void **) &d_x, sizeof(float) * points.count);
  cudaMemcpy(d_x, points.x, sizeof(float) * points.count, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_y, sizeof(float) * points.count);
  cudaMemcpy(d_y, points.y, sizeof(float) * points.count, cudaMemcpyHostToDevice);

  // Define hyperparameters
  float learning_rate = 0.0001;

  // y = mx + b
  float initial_b[1] = {0};
  float initia_m[1] = {0};

  int number_of_iterations = 1000;

  // Calculate the inital error
  float error = calculate_error(initial_b[0], initia_m[0], d_x, d_y, points.count);
  printf("start gradient descent at b = %f, m = %f, error = %f\n", initial_b[0], initia_m[0], error);

  // Perform gradient descent
  gradient_descent(initial_b, initia_m, d_x, d_y, points.count,
    number_of_iterations, learning_rate);

  // Calculate the new error
  error = calculate_error(initial_b[0], initia_m[0], d_x, d_y, points.count);
  printf("end point at b = %f, m = %f, error = %f\n", initial_b[0], initia_m[0], error);

  cudaFree(d_x);
  cudaFree(d_y);

  free(points.x);
  free(points.y);
}

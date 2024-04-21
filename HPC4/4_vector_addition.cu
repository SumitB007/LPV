%%cu
#include <iostream>
using namespace std;

__global__ void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}


void initialize(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 10;
    }
}

void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

int main() {
    int N = 4;
    int* A, * B, * C;

    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);

    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];

    initialize(A, vectorSize);
    initialize(B, vectorSize);

    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);

    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);

    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);

    cout << "Addition: ";
    print(C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}



// This CUDA C++ code performs vector addition using GPU parallelism. Let's go through the code step by step:

// 1. **Kernel Function (`add`)**:
//    - The kernel function is responsible for adding corresponding elements of two input vectors `A` and `B` and storing the result in vector `C`.
//    - Each thread is assigned a unique thread ID (`tid`) calculated based on the block index and thread index.
//    - Each thread performs the addition operation for one element of the vectors, checking first if the thread ID is within the bounds of the vectors.

// 2. **Helper Functions (`initialize` and `print`)**:
//    - `initialize`: This function initializes a vector with random values between 0 and 9. It takes a pointer to the vector and its size as arguments.
//    - `print`: This function prints the elements of a vector. It takes a pointer to the vector and its size as arguments.

// 3. **Main Function**:
//    - The main function initializes vectors `A` and `B`, prints them, and then performs vector addition using CUDA.
//    - Vectors `A` and `B` are initialized with random values using the `initialize` function and printed using the `print` function.
//    - Device memory (`X`, `Y`, and `Z`) is allocated using `cudaMalloc` for vectors `A`, `B`, and `C` respectively.
//    - The values of vectors `A` and `B` are copied from host to device memory using `cudaMemcpy`.
//    - The number of threads per block (`threadsPerBlock`) is set to 256, and the number of blocks per grid (`blocksPerGrid`) is calculated based on the size of the vectors.
//    - The kernel function `add` is launched with the specified number of blocks and threads per block.
//    - The result vector `C` is copied back from device to host memory using `cudaMemcpy` and printed.
//    - Finally, memory allocated on the device is freed using `cudaFree`.

// 4. **Output**:
//    - The code prints vectors `A` and `B` before addition, and the result vector `C` after addition.

// 5. **Memory Management**:
//    - Memory allocated on the device (`X`, `Y`, and `Z`) is freed at the end of the main function to release GPU resources.

// This code demonstrates how to leverage GPU parallelism using CUDA to perform vector addition, which can significantly accelerate computation for large vectors compared to sequential CPU-based computation.
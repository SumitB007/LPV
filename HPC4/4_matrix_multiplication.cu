%%cu
#include <iostream>
using namespace std;


// CUDA code to multiply matrices
__global__ void multiply(int* A, int* B, int* C, int size) {
    // Uses thread indices and block indices to compute each element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}


void initialize(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10;
    }
}


void print(int* matrix, int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            cout << matrix[row * size + col] << " ";
        }
        cout << '\n';
    }
    cout << '\n';
}


int main() {
    int* A, * B, * C;

    int N = 2;
    int blockSize =  16;

    int matrixSize = N * N;
    size_t matrixBytes = matrixSize * sizeof(int);

    A = new int[matrixSize];
    B = new int[matrixSize];
    C = new int[matrixSize];

    initialize(A, N);
    initialize(B, N);
    cout << "Matrix A: \n";
    print(A, N);

    cout << "Matrix B: \n";
    print(B, N);

    
    int* X, * Y, * Z;
    // Allocate space
    cudaMalloc(&X, matrixBytes);
    cudaMalloc(&Y, matrixBytes);
    cudaMalloc(&Z, matrixBytes);

    // Copy values from A to X
    cudaMemcpy(X, A, matrixBytes, cudaMemcpyHostToDevice);
    
    // Copy values from A to X and B to Y
    cudaMemcpy(Y, B, matrixBytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 2;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    multiply<<<blocks, threads>>>(X, Y, Z, N);

    cudaMemcpy(C, Z, matrixBytes, cudaMemcpyDeviceToHost);
    cout << "Multiplication of matrix A and B: \n";
    print(C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}



// This CUDA C++ code demonstrates matrix multiplication using GPU parallelism. Let's break down the code:

// 1. **Kernel Function (`multiply`)**:
//    - This kernel function is responsible for computing the product of two matrices `A` and `B`.
//    - It takes pointers to the input matrices `A` and `B`, as well as the output matrix `C`, and the size of the matrices (`size`).
//    - Each thread computes one element of the output matrix `C`.
//    - The thread indices (`row` and `col`) are computed using block and thread indices, and each thread iterates over the corresponding row of matrix `A` and column of matrix `B` to calculate the dot product.

// 2. **Helper Functions (`initialize` and `print`)**:
//    - `initialize`: This function initializes a matrix with random values between 0 and 9. It takes a pointer to the matrix and its size as arguments.
//    - `print`: This function prints the elements of a matrix. It takes a pointer to the matrix and its size as arguments.

// 3. **Main Function**:
//    - The main function initializes matrices `A` and `B`, prints them, and then performs matrix multiplication using CUDA.
//    - Matrices `A` and `B` are initialized with random values using the `initialize` function and printed using the `print` function.
//    - Device memory (`X`, `Y`, and `Z`) is allocated using `cudaMalloc` for matrices `A`, `B`, and `C` respectively.
//    - The values of matrices `A` and `B` are copied from host to device memory using `cudaMemcpy`.
//    - The number of threads per block (`THREADS`) is set to 2, and the number of blocks per grid (`BLOCKS`) is calculated based on the size of the matrices.
//    - The kernel function `multiply` is launched with the specified number of blocks and threads per block.
//    - The result matrix `C` is copied back from device to host memory using `cudaMemcpy` and printed.
//    - Finally, memory allocated on the device is freed using `cudaFree`.

// 4. **Output**:
//    - The code prints the matrices `A` and `B` before multiplication, and the result matrix `C` after multiplication.

// 5. **Memory Management**:
//    - Memory allocated on the device (`X`, `Y`, and `Z`) is freed at the end of the main function to release GPU resources.

// Overall, this code demonstrates how to leverage GPU parallelism using CUDA to perform matrix multiplication, which can significantly accelerate computation for large matrices compared to sequential CPU-based computation.
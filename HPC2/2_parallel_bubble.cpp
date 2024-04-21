#include <omp.h>
#include <stdlib.h>

#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using namespace std;

void s_bubble(int *, int);
void p_bubble(int *, int);
void swap(int &, int &);

void s_bubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void p_bubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;
    #pragma omp parallel for shared(a, first) num_threads(16)
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void swap(int &a, int &b) {
    int test;
    test = a;
    a = b;
    b = test;
}

std::string bench_traverse(std::function<void()> traverse_fn) {
    auto start = high_resolution_clock::now();
    traverse_fn();
    auto stop = high_resolution_clock::now();

    // Subtract stop and start timepoints and cast it to required unit.
    // Predefined units are nanoseconds, microseconds, milliseconds, seconds,
    // minutes, hours. Use duration_cast() function.
    auto duration = duration_cast<milliseconds>(stop - start);

    // To get the value of duration use the count() member function on the
    // duration object
    return std::to_string(duration.count());
}

int main(int argc, const char **argv) {
    if (argc < 3) {
        std::cout << "Specify array length and maximum random value\n";
        return 1;
    }
    int *a, n, rand_max;

    n = stoi(argv[1]);
    rand_max = stoi(argv[2]);
    a = new int[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand() % rand_max;
    }

    int *b = new int[n];
    copy(a, a + n, b);
    cout << "Generated random array of length " << n << " with elements between 0 to " << rand_max
         << "\n\n";

    std::cout << "Sequential Bubble sort: " << bench_traverse([&] { s_bubble(a, n); }) << "ms\n";
    // cout << "Sorted array is =>\n";
    // for (int i = 0; i < n; i++) {
    //     cout << a[i] << ", ";
    // }
    // cout << "\n\n";

    omp_set_num_threads(16);
    std::cout << "Parallel (16) Bubble sort: " << bench_traverse([&] { p_bubble(b, n); }) << "ms\n";
    // cout << "Sorted array is =>\n";
    // for (int i = 0; i < n; i++) {
    //     cout << b[i] << ", ";
    // }
    return 0;
}

/*

OUTPUT:
Generated random array of length 100 with elements between 0 to 200

Sequential Bubble sort: 0ms
Sorted array is =>
2, 3, 8, 11, 11, 12, 13, 14, 21, 21, 22, 26, 26, 27, 29, 29, 34, 42, 43, 46, 49, 51, 56, 57, 58, 59,
60, 62, 62, 67, 69, 73, 76, 76, 81, 84, 86, 87, 90, 91, 92, 94, 95, 105, 105, 113, 115, 115, 119,
123, 124, 124, 125, 126, 126, 127, 129, 129, 130, 132, 135, 135, 136, 136, 137, 139, 139, 140, 145,
150, 154, 156, 162, 163, 164, 167, 167, 167, 168, 168, 170, 170, 172, 173, 177, 178, 180, 182, 182,
183, 184, 184, 186, 186, 188, 193, 193, 196, 198, 199,

Parallel (16) Bubble sort: 1ms
Sorted array is =>
2, 3, 8, 11, 11, 12, 13, 14, 21, 21, 22, 26, 26, 27, 29, 29, 34, 42, 43, 46, 49, 51, 56, 57, 58, 59,
60, 62, 62, 67, 69, 73, 76, 76, 81, 84, 86, 87, 90, 91, 92, 94, 95, 105, 105, 113, 115, 115, 119,
123, 124, 124, 125, 126, 126, 127, 129, 129, 130, 132, 135, 135, 136, 136, 137, 139, 139, 140, 145,
150, 154, 156, 162, 163, 164, 167, 167, 167, 168, 168, 170, 170, 172, 173, 177, 178, 180, 182, 182,
183, 184, 184, 186, 186, 188, 193, 193, 196, 198, 199,


OUTPUT:

Generated random array of length 100000 with elements between 0 to 100000

Sequential Bubble sort: 16878ms
Parallel (16) Bubble sort: 2914ms




This code implements both sequential and parallel versions of the Bubble Sort algorithm using OpenMP for parallelization. Let's break down the code:

1. **Header Includes**: The code includes necessary header files such as `<omp.h>` for OpenMP, `<stdlib.h>` for standard library functions, `<array>` for using arrays, `<chrono>` for measuring time, `<functional>` for using `std::function`, `<iostream>` for input/output operations, `<string>` for string manipulation, and `<vector>` for using vectors.

2. **Namespace**: The code uses the `std` namespace for standard library functions and types.

3. **Bubble Sort Functions**:
   - `s_bubble`: This function implements the sequential version of the Bubble Sort algorithm. It iterates through the array and swaps adjacent elements if they are in the wrong order.
   - `p_bubble`: This function implements the parallel version of the Bubble Sort algorithm using OpenMP. It parallelizes the inner loop of the Bubble Sort algorithm, where elements are compared and swapped. OpenMP directives are used to distribute the loop iterations across multiple threads.

4. **Swap Function**: The `swap` function is a utility function used to swap two integers.

5. **Benchmarking Function**: The `bench_traverse` function takes a function object `traverse_fn`, executes it, measures the execution time, and returns the time taken in milliseconds.

6. **Main Function**:
   - It checks if the command-line arguments specifying the array length and maximum random value are provided. If not, it prints an error message and exits.
   - It initializes two arrays `a` and `b` of length `n` with random integers.
   - It measures the execution time of the sequential Bubble Sort algorithm using `s_bubble` and prints the time taken.
   - It measures the execution time of the parallel Bubble Sort algorithm using `p_bubble` and prints the time taken.

7. **Output**: The code prints the execution time of both sequential and parallel Bubble Sort algorithms. By commenting out the loops printing sorted arrays, it doesn't print the sorted arrays themselves.

Overall, this code demonstrates how to implement and compare the performance of sequential and parallel Bubble Sort algorithms using OpenMP.

g++ -fopenmp ./2_parallel_bubble.cpp -o ./program
.\program.exe 100 100
*/
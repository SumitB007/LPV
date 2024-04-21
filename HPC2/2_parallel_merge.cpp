#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; ++i) {
        L[i] = arr[l + i];
    }
    for (int j = 0; j < n2; ++j) {
        R[j] = arr[m + 1 + j];
    }

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1) {
        arr[k++] = L[i++];
    }
    while (j < n2) {
        arr[k++] = R[j++];
    }
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);
            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

int main() {
    const int n = 100000; // Change the array size as needed
    vector<int> arr(n), arrCopy(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % n;
        arrCopy[i] = arr[i];
    }

    auto start = high_resolution_clock::now();
    mergeSort(arr, 0, n - 1);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Sequential Merge Sort Time: " << duration.count() << "ms" << endl;

    // cout << "Sorted array after sequential merge sort:" << endl;
    // for (int i = 0; i < n; ++i) {
    //     cout << arr[i] << " ";
    // }
    cout << endl;

    start = high_resolution_clock::now();
    parallelMergeSort(arrCopy, 0, n - 1);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Parallel Merge Sort Time: " << duration.count() << "ms" << endl;

    // cout << "Sorted array after parallel merge sort:" << endl;
    // for (int i = 0; i < n; ++i) {
    //     cout << arrCopy[i] << " ";
    // }
    cout << endl;

    return 0;
}



/*
This code implements the merge sort algorithm using OpenMP for parallelization. Here's a breakdown of the code:

1. **Header Includes**: The code includes necessary header files such as `<omp.h>` for OpenMP, `<iostream>` for input/output operations, `<vector>` for using vectors, `<cstdlib>` for random number generation, and `<chrono>` for measuring time.

2. **Namespace**: The code uses the `std` namespace for standard library functions and types.

3. **Merge Function**: The `merge` function takes a vector `arr` and three indices `l`, `m`, and `r`, representing the left, middle, and right indices of the subarray to be merged. It merges the two subarrays `arr[l..m]` and `arr[m+1..r]` into a single sorted array.

4. **Merge Sort Function**: The `mergeSort` function recursively sorts the given vector `arr` in ascending order using the merge sort algorithm. It divides the array into two halves, recursively sorts the two halves, and then merges the sorted halves.

5. **Parallel Merge Sort Function**: The `parallelMergeSort` function is similar to `mergeSort`, but it utilizes OpenMP for parallelization. It splits the array into two halves, and then parallelizes the sorting of each half using OpenMP sections. After sorting each half in parallel, it merges the two sorted halves.

6. **Main Function**: In the `main` function:
   - It initializes a vector `arr` of size `n` with random integers and makes a copy of it named `arrCopy`.
   - It measures the execution time of sequential merge sort by calling `mergeSort` and prints the time taken.
   - It measures the execution time of parallel merge sort by calling `parallelMergeSort` and prints the time taken.

7. **Output**: The code prints the execution time of both sequential and parallel merge sort algorithms. By commenting out the loops printing sorted arrays, it doesn't print the sorted arrays themselves.

Overall, this code demonstrates how to implement and compare the performance of sequential and parallel merge sort algorithms using OpenMP.

*/
/*
  Windows does not support user defined reductions.
  This program may not run on MVSC++ compilers for Windows.
  Please use Linux Environment.[You can try using "windows subsystem for linux"]
*/

#include<iostream>
#include<omp.h>

using namespace std;
int minval(int arr[], int n){
  int minval = arr[0];
  #pragma omp parallel for reduction(min : minval)
    for(int i = 0; i < n; i++){
      if(arr[i] < minval) minval = arr[i];
    }
  return minval;
}

int maxval(int arr[], int n){
  int maxval = arr[0];
  #pragma omp parallel for reduction(max : maxval)
    for(int i = 0; i < n; i++){
      if(arr[i] > maxval) maxval = arr[i];
    }
  return maxval;
}

int sum(int arr[], int n){
  int sum = 0;
  #pragma omp parallel for reduction(+ : sum)
    for(int i = 0; i < n; i++){
      sum += arr[i];
    }
  return sum;
}

int average(int arr[], int n){
  return (double)sum(arr, n) / n;
}

int main(){
  int n = 5;
  int arr[] = {1,2,3,4,5};
  cout << "The minimum value is: " << minval(arr, n) << '\n';
  cout << "The maximum value is: " << maxval(arr, n) << '\n';
  cout << "The summation is: " << sum(arr, n) << '\n';
  cout << "The average is: " << average(arr, n) << '\n';
  return 0;
}


/*
This C++ code demonstrates how to use OpenMP directives for parallelization to compute various statistics such as minimum value, maximum value, sum, and average of an array concurrently.

Let's break down the code:

1. **Header Includes**:
   - `#include<iostream>`: This header file includes declarations for the input/output stream objects like `cout` and `cin`.
   - `#include<omp.h>`: This header file provides OpenMP directives and functions for parallel programming.

2. **Namespace**:
   - `using namespace std;`: This line allows the usage of standard library functions and objects without specifying the `std::` prefix.

3. **Function Definitions**:
   - `minval`: This function calculates the minimum value in an array using OpenMP parallelization. It initializes `minval` with the first element of the array and then uses a parallel loop with the reduction clause to find the minimum value across all elements.
   - `maxval`: Similar to `minval`, this function calculates the maximum value in an array using parallelization.
   - `sum`: This function computes the sum of all elements in the array using parallelization.
   - `average`: This function calculates the average value of elements in the array by dividing the sum by the number of elements.

4. **Main Function**:
   - It initializes an array `arr` with 5 integers.
   - It calls the `minval`, `maxval`, `sum`, and `average` functions to compute the respective statistics and prints the results using `cout`.

5. **Parallelization**:
   - OpenMP directives are used within the `minval`, `maxval`, and `sum` functions to parallelize the loops.
   - The `reduction` clause is used in each parallel loop to perform the reduction operation (e.g., `min`, `max`, `+`) on a variable (`minval`, `maxval`, `sum`) across all threads.

6. **Output**:
   - The code prints the minimum value, maximum value, sum, and average of the array.

Overall, this code demonstrates how to utilize OpenMP parallelization to efficiently compute statistics for an array in a concurrent manner.

*/
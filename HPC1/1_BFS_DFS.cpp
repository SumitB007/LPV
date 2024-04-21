#include <iostream>
#include <atomic>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Graph class representing the adjacency list
class Graph {
    int V;  // Number of vertices
    vector<vector<int>> adj;  // Adjacency list

public:
    Graph(int V) : V(V), adj(V) {}

    // Add an edge to the graph
    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }

    
    // Parallel DFS utility function
    void parallelDFSUtil(int v, vector<atomic<bool>>& visited) {
        visited[v] = true;
        cout << v << " ";

        #pragma omp parallel for
        for (int i = 0; i < adj[v].size(); ++i) {
            int n = adj[v][i];
            if (!visited[n])
                parallelDFSUtil(n, visited);
        }
    }

    // Parallel Depth-First Search
    void parallelDFS(int startVertex) {
        vector<atomic<bool>> visited(V);
        for(int i=0; i<V; ++i)
            visited[i] = false;
        parallelDFSUtil(startVertex, visited);
    }

    // Parallel Breadth-First Search
    void parallelBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[startVertex] = true;
        q.push(startVertex);

        while (!q.empty()) {
            int v = q.front();
            q.pop();
            cout << v << " ";

            #pragma omp parallel for
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                if (!visited[n]) {
                    visited[n] = true;
                    q.push(n);
                }
            }
        }
    }
};


int main() {
    // Create a graph
    Graph g(7);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    
    /*
        0 -------->1
        |         / \
        |        /   \
        |       /     \
        v       v       v
        2       3       4
        |      |
        |      |
        v      v
        5      6
    */

    cout << "Depth-First Search (DFS): ";
    g.parallelDFS(0);
    cout << endl;

    cout << "Breadth-First Search (BFS): ";
    g.parallelBFS(0);
    cout << endl;

    return 0;
}


/*
This C++ code is a parallel implementation of the Depth-First Search (DFS) and Breadth-First Search (BFS) 
algorithms for traversing or searching graph data structures. It uses OpenMP, a library for parallel programming 
in C++, to parallelize the algorithms.

Here's a breakdown of how the code works:

The Graph class represents a graph using an adjacency list. It has a constructor that initializes the number of vertices (V) 
and the adjacency list (adj), and a method addEdge to add an edge between two vertices.

The parallelDFSUtil method is a helper function for the parallel DFS algorithm. It marks a vertex as visited and then 
recursively visits all its unvisited neighbors. The loop that iterates over the neighbors is parallelized using 
the #pragma omp parallel for directive.

The parallelDFS method initializes a vector of atomic booleans to keep track of the visited vertices and then 
calls parallelDFSUtil on the start vertex. The use of std::atomic<bool> ensures that updates to the visited 
vector are thread-safe

The parallelBFS method implements the parallel BFS algorithm. It uses a queue to keep track of vertices to visit
 and a vector of booleans to keep track of visited vertices. The loop that iterates over the neighbors of a vertex 
 is parallelized using the #pragma omp parallel for directive.


 The main function creates a Graph object, adds edges to it to form a specific graph, and then calls parallelDFS 
 and parallelBFS to perform DFS and BFS on the graph, starting from vertex 0.


The libraries used in this code are:

<iostream>: This is a standard C++ library for input/output operations. It's used here to print the order of 
vertices visited by the DFS and BFS algorithms 

<atomic>: This is a C++ library that provides components for fine-grained atomic operations,
 allowing for lock-free concurrent programming. It's used here to ensure that updates to the visited vector 
 in the parallel DFS algorithm are thread-safe.


<vector>: This is a standard C++ library that provides a dynamic array. It's used here to implement the adjacency
 list and the visited vector.

<queue>: This is a standard C++ library that provides a queue data structure. It's used here in the BFS algorithm 
to keep track of vertices to visit.

<omp.h>: This is the header file for OpenMP, a library for parallel programming in C++. It's used here to parallelize
the DFS and BFS algorithms



Command to run:

#1

g++ -fopenmp .\1_BFS_DFS.cpp -o ./output

./output.exe

#2
g++ -fopenmp .\1_BFS_DFS.cpp
./a.out
*/
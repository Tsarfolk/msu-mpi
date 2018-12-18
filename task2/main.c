#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "mpi.h"
#include "omp.h"

double Lx = M_PI;
double Ly = M_PI;
double Lz = M_PI;

typedef struct CPoint XYZ;

struct CPoint {
    int x, y, z;
};

XYZ init(int x, int y, int z) {
    XYZ entity;
    entity.x = x;
    entity.y = y;
    entity.z = z;
    return entity;
}

int factor_of_2(int num) {
    int count = 0;
    while (true) {
        num /= 2;
        if (num == 0) { return count; }
        count += 1;
    }
    
    return count;
}

void distribution() {
    
}

void start(int argc, char * argv[], int *size, int *rank) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void syncThreads() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void finilize() {
    MPI_Finalize();
}

int main(int argc, char * argv[]) {
    int size, rank;
    start(argc, argv, &size, &rank);
    syncThreads();
    printf("%d %d", size, rank);
    
    finilize();
    return 0;
}

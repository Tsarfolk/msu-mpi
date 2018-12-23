#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "mpi.h"
#include "omp.h"

typedef struct CPoint XYZ;
typedef struct APoint AXYZ;
typedef struct FPoint FXYZ;

struct CPoint {
    int x, y, z;
};

struct APoint {
    double *x, *y, *z;
};

struct FPoint {
    double x, y, z;
};

FXYZ L;
double k[3] = {6.25, 4 * M_PI, 6 * M_PI};
double kNorm = 0;
double tau = 0.001;

double ut(FXYZ point, double t) {
    return cos(kNorm * t) * cos(k[0] * point.x + k[1] * point.y + k[2] * point.z);
}

double u0(FXYZ point){
    return cos(k[0] * point.x + k[1] * point.y + k[2] * point.z);
}

FXYZ finit(double x, double y, double z) {
    FXYZ entity;
    entity.x = x;
    entity.y = y;
    entity.z = z;
    return entity;
}

XYZ init(int x, int y, int z) {
    XYZ entity;
    entity.x = x;
    entity.y = y;
    entity.z = z;
    return entity;
}

double fvalueAt(int index, FXYZ item) {
    double array[3] = {item.x, item.y, item.z};
    return array[index];
}

int valueAt(int index, XYZ item) {
    int array[3] = {item.x, item.y, item.z};
    return array[index];
}

void setValueAt(int index, int value, XYZ *item) {
    int *array[3] = {&item->x, &item->y, &item->z};
    *array[index] = value;
}

int addValueAt(int index, int value, XYZ *item) {
    int *array[3] = {&item->x, &item->y, &item->z};
    *array[index] += value;
    return *array[index];
}

int scalar(XYZ a, XYZ b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void print(XYZ item) { printf("XYZ is %d %d %d\n", item.x, item.y, item.z); }
void fprint(FXYZ item) { printf("FXYZ is %lf %lf %lf\n", item.x, item.y, item.z); }

void calcualteDistribution(XYZ *range, XYZ *rankMultiplier, int proccessCount) {
    int proccessFactor = 0;
    while (true) {
        proccessCount /= 2;
        if (proccessCount == 0) { break; }
        proccessFactor += 1;
    }
    int factor = proccessFactor % 3 == 2 ? proccessFactor / 3 + 1 : proccessFactor / 3;
    
    range->x = pow(2, factor);
    range->y = pow(2, factor);
    range->z = pow(2, proccessFactor - 2 * factor);
    rankMultiplier->x = 1;
    rankMultiplier->y = range->x;
    rankMultiplier->z = range->x * range->y;
}

void convertRankToPoint(int rank, XYZ range, XYZ *point) {
    point->x = rank % range.x;
    rank /= range.x;
    point->y = rank % range.y;
    rank /= range.y;
    point->z = rank;
}

int convertPointToRank(XYZ range, XYZ rankMultiplier, XYZ point) {
    int rank = 0;
    rank += rankMultiplier.x * ((point.x + range.x) % range.x);
    rank += rankMultiplier.y * ((point.y + range.y) % range.y);
    rank += rankMultiplier.z * ((point.z + range.z) % range.z);
    return rank;
}

void start(int argc, char * argv[], int *processCount, int *rank) {
    if (argc >= 1) {
        *processCount = atoi(argv[1]);
    } else {
        *processCount = 16;
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void syncThreads() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void startTimer(double *time) {
    *time = MPI_Wtime();
}

void finilize() {
    MPI_Finalize();
}

void calculateU(AXYZ *u, XYZ uSize, XYZ dotsNumber, FXYZ baseCoordinate, FXYZ step) {
    int memorySize = uSize.x * uSize.y * uSize.z;
    u->x = malloc(memorySize * sizeof(double));
    u->y = malloc(memorySize * sizeof(double));
    u->z = malloc(memorySize * sizeof(double));
    
    XYZ point;
    
    for (point.x = 0; point.x < dotsNumber.x; point.x += 1) {
        for (point.y = 0; point.y < dotsNumber.y; point.y += 1) {
            for (point.z = 0; point.z < dotsNumber.z; point.z += 1) {
                long long int index = scalar(point, uSize);
                FXYZ nextPoint = finit(baseCoordinate.x + step.x * point.x,
                                       baseCoordinate.y + step.y * point.y,
                                       baseCoordinate.z + step.z * point.z);
                u->x[index] = u0(nextPoint);
            }
        }
    }
}

void initIteratorParams(int gridSize, FXYZ *step, FXYZ *baseCoordinate, XYZ *dotsNumber, XYZ blockCoordinate, XYZ range) {
    double intervals = gridSize - 1;
    *dotsNumber = init(gridSize / range.x,
                       gridSize / range.y,
                       gridSize / range.z);
    *step = finit(L.x / intervals,
                  L.y / intervals,
                  L.z / intervals);
    *baseCoordinate = finit(step->x * dotsNumber->x * blockCoordinate.x,
                            step->y * dotsNumber->y * blockCoordinate.y,
                            step->z * dotsNumber->z * blockCoordinate.z);
}

void parseCLIParams(char * argv[], int argc, int *gridSize, double *gridSteps) {
    if (argc >= 3) {
        *gridSize = atoi(argv[1]);
        *gridSteps = atof(argv[2]);
    } else {
        *gridSize = 64;
        *gridSteps = 0.03;
    }
}

void processCalculation(XYZ dotsNumber, XYZ blockCoordinate, XYZ range, XYZ uSize, AXYZ u, XYZ rankMultiplier, FXYZ step, FXYZ baseCoordinate, int rank, double T_fin, int processCount, double executionTime) {
    printf("[182] My rank is %d", rank);
    int count = 2 * 6;
    MPI_Request *request = malloc(count * sizeof(MPI_Request));
    
    double **send = malloc(6 * sizeof(double *));
    double **recv = malloc(6 * sizeof(double *));
    
    int index_inf[6] = {1, 2, 0, 2, 0, 1};
    int **srHelper = malloc(3 * sizeof(int *));
    for(int i = 0; i < 3; ++i){
        srHelper[i] = malloc(3 * sizeof(int));
        srHelper[i][0] = 1;
    }
    srHelper[0][1] = dotsNumber.y;
    srHelper[1][1] = dotsNumber.x;
    srHelper[2][1] = dotsNumber.x;
    srHelper[0][2] = dotsNumber.y * dotsNumber.z;
    srHelper[1][2] = dotsNumber.x * dotsNumber.z;
    srHelper[2][2] = dotsNumber.x * dotsNumber.y;
    

    // alloc
    short squareTopology[6] = {1, 1, 1, 1, 1, 1};
    short cond[3] = {1, 0, 1};
    for (int i = 0; i < 6; ++i) {
        if(valueAt(i / 2, blockCoordinate) == (i % 2 == 0 ? 0 : valueAt(i / 2, range) - 1)){
            if (cond[i / 2] == 1){
                squareTopology[i] -= 1;
            } else {
                squareTopology[i] += 1;
            }
        }
        send[i] = malloc(squareTopology[i] * srHelper[i / 2][2] * sizeof(double));
        recv[i] = malloc(squareTopology[i] * srHelper[i / 2][2] * sizeof(double));
    }

    for(int i = 0; i < 6; i += 1) {
        int i0 = i / 2, i1 = index_inf[2 * (i / 2)], i2 = index_inf[2 * (i / 2) + 1];
        int st = (i % 2 == 0 ? 0 : valueAt(i0, dotsNumber) - squareTopology[i]);
        int ed = (i % 2 == 0 ? squareTopology[i] : valueAt(i0, dotsNumber));
        XYZ point;
        int dotsI1 = valueAt(i1, dotsNumber);
        int dotsI2 = valueAt(i2, dotsNumber);
        for(setValueAt(i0, st, &point);  valueAt(i0, point) < ed; addValueAt(i0, 1, &point)) {
            for(setValueAt(i1, 0, &point);  valueAt(i1, point) < dotsI1; addValueAt(i1, 1, &point)) {
                for(setValueAt(i2, 0, &point);  valueAt(i2, point) < dotsI2; addValueAt(i2, 1, &point)) {
                    long long int ind = scalar(point, uSize);
                    int multiplier = (i % 2 == 0 ? valueAt(i0, point) : valueAt(i0, point) - ed + squareTopology[i]);
                    int sendIndex = valueAt(i1, point) * srHelper[i0][0] + valueAt(i2, point) * srHelper[i0][1] + multiplier * srHelper[i0][2];
                    send[i][sendIndex] = u.x[ind];
                }
            }
        }
    }
    
    // bound points exchange
    int cnt = 0;
    for(int i = 0; i < 6; ++i) {
        if (squareTopology[i] > 0) {
        	int value = (i % 2 == 0) ? -1 : 1;
        	addValueAt(i / 2, value, &blockCoordinate);
            if (convertPointToRank(range, rankMultiplier, blockCoordinate) == rank) {
                double *tmp = send[i];
                send[i] = recv[i % 2 == 0 ? i + 1 : i - 1];
                recv[i % 2 == 0 ? i + 1 : i - 1] = tmp;
            } else {
                MPI_Send_init(send[i], squareTopology[i] * srHelper[i / 2][2], MPI_DOUBLE, convertPointToRank(range, rankMultiplier, blockCoordinate), i % 2, MPI_COMM_WORLD, &(request[cnt]));
                cnt += 1;
                MPI_Recv_init(recv[i], squareTopology[i] * srHelper[i / 2][2], MPI_DOUBLE, convertPointToRank(range, rankMultiplier, blockCoordinate), (i + 1) % 2, MPI_COMM_WORLD, &(request[cnt]));
                cnt += 1;
            }
            addValueAt(i / 2, (-1) * value, &blockCoordinate);
        }
    }
    
    // sync exchange
    MPI_Startall(cnt, request);
    MPI_Waitall(cnt, request, MPI_STATUSES_IGNORE);

    // first approximation
    XYZ point;
    for (point.x = 0; point.x < dotsNumber.x; point.x += 1) {
        for (point.y = 0; point.y < dotsNumber.y; point.y += 1) {
            for (point.z = 0; point.z < dotsNumber.z; point.z += 1) {
                double delta = 0, d[3];
                long long int ind;
                bool f = false;
                for(int i = 2; i >= 0; i -= 1) {
                    double previous = 0;
                    double current = 0;
                    double next = 0;
                    ind = scalar(point, uSize);
                    current = u.x[ind];
                    if (valueAt(i, point) == 0) {
                        if (squareTopology[2 * i] == 0) {
                            current = 0;
                            f = true;
                        }

                        int index = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                        if (squareTopology[2 * i] == 1) {
                            previous = recv[2 * i][index];
                        } else if (squareTopology[2 * i] == 2) {
                            previous = recv[2 * i][index];
                            current = recv[2 * i][index + srHelper[i][2]];
                        }
                    } else {
                        addValueAt(i, -1, &point);
                        ind = scalar(point, uSize);
                        previous = u.x[ind];
                        addValueAt(i, 1, &point);
                    }
                    if (valueAt(i, point) == valueAt(i, dotsNumber) - 1) {
                        if (squareTopology[2 * i + 1] == 0){
                            next = -previous; current = 0;
                            f = true;
                        }
                        int index = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                        if (squareTopology[2 * i + 1] == 1){
                            next = recv[2 * i + 1][index];
                        }
                        if (squareTopology[2 * i + 1] == 2){
                            current = recv[2 * i + 1][index];
                            next = recv[2 * i + 1][index + srHelper[i][2]];
                        }
                    } else {
                        addValueAt(i, 1, &point);
                        ind = scalar(point, uSize);
                        next = u.x[ind];
                        addValueAt(i, -1, &point);
                    }
                    if (valueAt(i, point) == 0 && squareTopology[2 * i] == 0) {
                        previous = -next;
                    }
                    d[i] = previous - 2 * current + next;
                    d[i] /= fvalueAt(i, step) * fvalueAt(i, step);

                    delta += d[i];
                }
                ind = scalar(point, uSize);

                if (f) {
                    FXYZ fpoint = finit(baseCoordinate.x + step.x * point.x,
                                        baseCoordinate.y + step.y * point.y,
                                        baseCoordinate.z + step.z * point.z);
                    u.y[ind] = ut(fpoint, tau);
                } else {
                    u.y[ind] = u.x[ind] + (tau * tau / 2) * delta;
                }
            }
        }
    }

    double errorRate = 0;
    double t = tau;
    while (t < T_fin) {
        for(int i = 0; i < 6; ++i){
            int i0 = i / 2, i1 = index_inf[2 * (i / 2)], i2 = index_inf[2 * (i / 2) + 1];
            int st = (i % 2 == 0 ? 0 : valueAt(i0, dotsNumber) - squareTopology[i]);
            int ed = (i % 2 == 0 ? squareTopology[i] : valueAt(i0, dotsNumber));
            XYZ point;
            for(setValueAt(i0, st, &point);  valueAt(i0, point) < ed; addValueAt(i0, 1, &point)) {
                for(setValueAt(i1, 0, &point);  valueAt(i1, point) < valueAt(i1, dotsNumber); addValueAt(i1, 1, &point)) {
                    for(setValueAt(i2, 0, &point);  valueAt(i2, point) < valueAt(i2, dotsNumber); addValueAt(i2, 1, &point)) {
                        long long int uIndex = scalar(point, uSize);
                        int mult = (i % 2 == 0 ? valueAt(i0, point) : valueAt(i0, point) - ed + squareTopology[i]);
                        int sendIndex = valueAt(i1, point) * srHelper[i0][0] + valueAt(i2, point) * srHelper[i0][1] + mult * srHelper[i0][2];
                        send[i][sendIndex] = u.y[uIndex];
                    }
                }
            }
        }
        
        int cnt = 0;
        for(int i = 0; i < 6; ++i){
            if (squareTopology[i] > 0){
            	int value = (i % 2 == 0) ? -1 : 1;
        		addValueAt(i / 2, value, &blockCoordinate);
                if (convertPointToRank(range, rankMultiplier, point) == rank) {
                    double *tmp = send[i];
                    send[i] = recv[i % 2 == 0 ? i + 1 : i - 1];
                    recv[i % 2 == 0 ? i + 1 : i - 1] = tmp;
                } else {
                    int rank = convertPointToRank(range, rankMultiplier, blockCoordinate);
                     MPI_Send_init(send[i], squareTopology[i] * srHelper[i / 2][2], MPI_DOUBLE, rank, i % 2, MPI_COMM_WORLD, &(request[cnt]));
                    cnt += 1;
                     MPI_Recv_init(recv[i], squareTopology[i] * srHelper[i / 2][2], MPI_DOUBLE, rank, (i + 1) % 2, MPI_COMM_WORLD, &(request[cnt]));
                    cnt += 1;
                }
                addValueAt(i / 2, -1 * value, &blockCoordinate);
            }
        }
        
        MPI_Startall(cnt, request);
        MPI_Waitall(cnt, request, MPI_STATUSES_IGNORE);

//        #pragma omp parallel for num_threads(2)
        for (point.x = 0; point.x < dotsNumber.x; point.x += 1) {
            for (point.y = 0; point.y < dotsNumber.y; point.y += 1) {
                for (point.z = 0; point.z < dotsNumber.z; point.z += 1) {
                    double delta = 0, d[3];
                    long long int ind;
                    bool f = false;
                    for(int i = 0; i < 3; ++i) {
                    	double previous = 0;
                    	double current = 0;
                    	double next = 0;
                        ind = scalar(point, uSize);
                        current = u.y[ind];

                        if (valueAt(i, point) == 0) {
                            if (squareTopology[2 * i] == 0) {
                                current = 0;
                                f = true;
                            }
                            long long int recvIndex = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                            if (squareTopology[2 * i] == 1) {
                                previous = recv[2 * i][recvIndex];
                            }
                            if (squareTopology[2 * i] == 2) {
                                previous = recv[2 * i][recvIndex];
                                current = recv[2 * i][recvIndex + srHelper[i][2]];
                            }
                        } else {
                            addValueAt(i, -1, &point);
                            ind = scalar(point, uSize);
                            previous = u.y[ind];
                            addValueAt(i, 1, &point);
                        }
                        if (valueAt(i, point) == valueAt(i, dotsNumber) - 1) {
                            if (squareTopology[2 * i + 1] == 0){
                                next = -previous; current = 0;
                                f = true;
                            }
                            long long int recvIndex = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                            if (squareTopology[2 * i + 1] == 1) {
                                next = recv[2 * i + 1][recvIndex];
                            }
                            if (squareTopology[2 * i + 1] == 2) {
                                current = recv[2 * i + 1][recvIndex];
                                next = recv[2 * i + 1][recvIndex + srHelper[i][2]];
                            }
                        } else {
                            addValueAt(i, 1, &point);
                            ind = scalar(point, uSize);
                            next = u.y[ind];
                            addValueAt(i, -1, &point);
                        }
                        if (valueAt(i, point) == 0 && squareTopology[2 * i] == 0) {
                            previous = -next;
                        }
                        d[i] = previous - 2 * current + next;
                        d[i] /= fvalueAt(i, step) * fvalueAt(i, step);

                        delta += d[i];
                    }
                    ind = scalar(point, uSize);
                    if (f) {
                        FXYZ fpoint = finit(baseCoordinate.x + step.x * point.x,
                                            baseCoordinate.y + step.y * point.y,
                                            baseCoordinate.z + step.z * point.z);
                        u.z[ind] = ut(fpoint, t + tau);
                    } else {
                        u.z[ind] = 2 * u.y[ind] - u.x[ind] + tau * tau * delta;
                    }
                }
            }
        }

        double *tmp = u.z;
        u.z = u.x;
        u.x = u.y;
        u.y = tmp;
        t += tau;
    }

    syncThreads();
    executionTime = MPI_Wtime() - executionTime;

    errorRate = 0;
    for(point.x = 0; point.x < dotsNumber.x; point.x += 1) {
        for(point.y = 0; point.y < dotsNumber.y; point.y += 1) {
            for(point.z = 0; point.z < dotsNumber.z; point.z += 1) {
                long long int ind = scalar(point, uSize);
                FXYZ fpoint = finit(baseCoordinate.x + step.x * point.x,
                                    baseCoordinate.y + step.y * point.y,
                                    baseCoordinate.z + step.z * point.z);
                double diff = u.y[ind] - ut(fpoint, t);
                errorRate += diff * diff;
            }
        }
    }

    if (rank == 0) {
        double *errorRates = calloc(processCount, sizeof(double));
        errorRates[0] = errorRate;
        MPI_Request *requets = malloc(processCount * sizeof(MPI_Request));
        for(int i = 1; i < processCount; ++i){
            MPI_Recv_init(&(errorRates[i]), 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(requets[i - 1]));
        }
        MPI_Startall(processCount - 1, requets);
        MPI_Waitall(processCount - 1, requets, MPI_STATUSES_IGNORE);
        errorRate = 0;
        for(int i = 0; i < processCount; ++i){
            errorRate += errorRates[i];
        }
        errorRate = sqrt(errorRate);
        printf("Error rate %lf; Time: %lf\n", errorRate, executionTime);
        free(requets);
    } else {
        printf("Error rate of %d is %lf", rank, errorRate);
        MPI_Send(&errorRate, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    free(u.x);
    free(u.y);
    free(u.z);
    for(int i = 0; i < 3; ++i) {
        free(srHelper[i]);
        free(send[2 * i]);
        free(send[2 * i + 1]);
        free(recv[2 * i]);
        free(recv[2 * i + 1]);
    }
    free(send);
    free(recv);
    free(srHelper);
}

int main(int argc, char * argv[]) {
    int processCount, rank = 10;
    double executionTime = 0;
    int gridSize;
    double gridSteps;
    AXYZ u;
    XYZ range, rankMultiplier, blockCoordinate, dotsNumber;
    FXYZ step, baseCoordinate;
    
    L = finit(1, 1, 1);
    kNorm = sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
    start(argc, argv, &processCount, &rank);
    parseCLIParams(argv, argc, &gridSize, &gridSteps);
    
    if (rank == 0) {
        printf("Process count is %d\n", processCount);
        printf("Grid size is %d\n", gridSize);
        printf("Tau %lf\n", gridSteps);
    }

    syncThreads();
    startTimer(&executionTime);

    calcualteDistribution(&range, &rankMultiplier, processCount);
    convertRankToPoint(rank, range, &blockCoordinate);
    initIteratorParams(gridSize, &step, &baseCoordinate, &dotsNumber, blockCoordinate, range);

    XYZ uSize = init(1, dotsNumber.x, dotsNumber.x * dotsNumber.y);
    calculateU(&u, uSize, dotsNumber, baseCoordinate, step);
    processCalculation(dotsNumber, blockCoordinate, range, uSize, u, rankMultiplier, step, baseCoordinate, rank, gridSteps, processCount, executionTime);
    finilize();
    return 0;
}

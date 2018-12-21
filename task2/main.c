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

void initIteratorParams(int gridSize, FXYZ *step, FXYZ *baseCoordinate, XYZ *dotsNumber, XYZ coordinate, XYZ range) {
    double intervals = gridSize - 1;
    *dotsNumber = init(gridSize / range.x,
                       gridSize / range.y,
                       gridSize / range.z);
    *step = finit(L.x / intervals,
                  L.y / intervals,
                  L.z / intervals);
    *baseCoordinate = finit(step->x * dotsNumber->x * coordinate.x,
                            step->y * dotsNumber->y * coordinate.y,
                            step->z * dotsNumber->z * coordinate.z);
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

void calculate(XYZ dotsNumber, XYZ coordinate, XYZ range, XYZ uSize, AXYZ u, XYZ rankMultiplier, FXYZ step, FXYZ baseCoordinate, int rank, double T_fin, int processCount, double executionTime) {
    int count = 2 * 2 * 3;
    MPI_Request *request = malloc(count * sizeof(MPI_Request));
    
    double **send = malloc(2 * 3 * sizeof(double *));
    double **recv = malloc(2 * 3 * sizeof(double *));
    
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
    
    short msg_l[2 * 3] = {1, 1, 1, 1, 1, 1}; // 101
    short cond[3] = {1, 0, 1};
    for (int i = 0; i < 2 * 3; ++i) {
        if(valueAt(i / 2, coordinate) == (i % 2 == 0 ? 0 : valueAt(i / 2, range) - 1)){
            if (cond[i / 2] == 1){
                --(msg_l[i]);  //1го рода
            }else{
                ++(msg_l[i]);  //переодические
            }
        }
        send[i] = malloc(msg_l[i] * srHelper[i / 2][2] * sizeof(double));
        recv[i] = malloc(msg_l[i] * srHelper[i / 2][2] * sizeof(double));
    }
    printf("Rank is %d, msg_l is %d %d %d %d %d %d\n", rank, msg_l[0],msg_l[1],msg_l[2],msg_l[3],msg_l[4],msg_l[5]);
    printf("rank is %d, point is (%d, %d, %d)", rank, coordinate.x, coordinate.y, coordinate.z);

    //подготовка массивов обмена данных
    for(int i = 0; i < 2 * 3; i += 1){
        int i0 = i / 2, i1 = index_inf[2 * (i / 2)], i2 = index_inf[2 * (i / 2) + 1];
        int st = (i % 2 == 0 ? 0 : valueAt(i0, dotsNumber) - msg_l[i]);
        int ed = (i % 2 == 0 ? msg_l[i] : valueAt(i0, dotsNumber));
        XYZ point;
        int dotsI1 = valueAt(i1, dotsNumber);
        int dotsI2 = valueAt(i2, dotsNumber);
        for(setValueAt(i0, st, &point);  valueAt(i0, point) < ed; addValueAt(i0, 1, &point)) {
            for(setValueAt(i1, 0, &point);  valueAt(i1, point) < dotsI1; addValueAt(i1, 1, &point)) {
                for(setValueAt(i2, 0, &point);  valueAt(i2, point) < dotsI2; addValueAt(i2, 1, &point)) {
                    long long int ind = scalar(point, uSize);
                    int multiplier = (i % 2 == 0 ? valueAt(i0, point) : valueAt(i0, point) - ed + msg_l[i]);
                    int sendIndex = valueAt(i1, point) * srHelper[i0][0] + valueAt(i2, point) * srHelper[i0][1] + multiplier * srHelper[i0][2];
                    send[i][sendIndex] = u.x[ind];
                }
            }
        }
    }
    
    int cnt = 0; //реальное колическво пересылок
    //инициализация пересылок
    for(int i = 0; i < 2 * 3; ++i) {
        printf("Rank is %d, msg_l is %d %d %d %d %d %d, i is %d\n", rank, msg_l[0],msg_l[1],msg_l[2],msg_l[3],msg_l[4],msg_l[5], i);
        if (msg_l[i] > 0) {
            (i % 2 == 0) ? addValueAt(i / 2, -1, &coordinate) : addValueAt(i / 2, 1, &coordinate);
            printf("xyz_to_rank is %d, my_rank is %d, point is (%d, %d, %d)\n", convertPointToRank(range, rankMultiplier, coordinate), rank, coordinate.x, coordinate.y, coordinate.z);
            if (convertPointToRank(range, rankMultiplier, coordinate) == rank) {
                //если блок отправляет данные сам себе, пересылки на самом деле не нужны
                double *tmp = send[i];
                send[i] = recv[i % 2 == 0 ? i + 1 : i - 1];
                recv[i % 2 == 0 ? i + 1 : i - 1] = tmp;
            } else {
                int calculatedRank = convertPointToRank(range, rankMultiplier, coordinate);
//                printf("Rank is %d, coordinate is (%d, %d, %d), calculated rank is %d\n", rank, , calculatedRank);
                MPI_Send_init(send[i], msg_l[i] * srHelper[i / 2][2], MPI_DOUBLE, calculatedRank, i % 2, MPI_COMM_WORLD, &(request[cnt]));
                cnt += 1;
                MPI_Recv_init(recv[i], msg_l[i] * srHelper[i / 2][2], MPI_DOUBLE, calculatedRank, (i + 1) % 2, MPI_COMM_WORLD, &(request[cnt]));
                cnt += 1;
            }
            (i % 2 == 0) ? addValueAt(i / 2, 1, &coordinate) : addValueAt(i / 2, -1, &coordinate);
        }
    }
//
    MPI_Startall(cnt, request);
    MPI_Waitall(cnt, request, MPI_STATUSES_IGNORE);
    
//    считаем первое приближение
    XYZ point;
    for (point.x = 0; point.x < dotsNumber.x; point.x += 1) {
        for (point.y = 0; point.y < dotsNumber.y; point.y += 1) {
            for (point.z = 0; point.z < dotsNumber.z; point.z += 1) {
                //внутри блока реализована аппроксимация оператора Лапласса
                double delta = 0, d[3];
                long long int ind;
                bool f = false;
                for(int i = 3 - 1; i > -1; --i) {
                    double ub = 0, uc = 0, uf = 0;
                    ind = scalar(point, uSize);
                    uc = u.x[ind];
                    if (valueAt(i, point) == 0) {
                        if (msg_l[2 * i] == 0) {
                            uc = 0;
                            f = true;
                        }

                        int index = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                        if (msg_l[2 * i] == 1) {
                            ub = recv[2 * i][index];
                        } else if (msg_l[2 * i] == 2) {
                            //ub = recv; uc = recv;
                            ub = recv[2 * i][index];
                            uc = recv[2 * i][index + srHelper[i][2]];
                        }
                    } else {
                        addValueAt(i, -1, &point);
                        ind = scalar(point, uSize);
                        ub = u.x[ind];
                        addValueAt(i, 1, &point);
                    }
                    if (valueAt(i, point) == valueAt(i, dotsNumber) - 1) {
                        if (msg_l[2 * i + 1] == 0){
                            uf = -ub; uc = 0;
                            f = true;
                        }
                        int index = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                        if (msg_l[2 * i + 1] == 1){
                            uf = recv[2 * i + 1][index];
                        }
                        if (msg_l[2 * i + 1] == 2){
                            //ub = recv; uc = recv;
                            uc = recv[2 * i + 1][index];
                            uf = recv[2 * i + 1][index + srHelper[i][2]];
                        }
                    } else{
                        addValueAt(i, 1, &point);
                        ind = scalar(point, uSize);
                        uf = u.x[ind];
                        addValueAt(i, -1, &point);
                    }
                    if (valueAt(i, point) == 0 && msg_l[2 * i] == 0) {
                        ub = -uf;
                    }
                    d[i] = ub - 2 * uc + uf;
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

    //вот где все веселье начиается
    double norm = 0;
    double t = tau;
    while (t < T_fin) {
        //обмен u^n
        //подготовка массивов обмена данных
        for(int i = 0; i < 2 * 3; ++i){
            int i0 = i / 2, i1 = index_inf[2 * (i / 2)], i2 = index_inf[2 * (i / 2) + 1];
            int st = (i % 2 == 0 ? 0 : valueAt(i0, dotsNumber) - msg_l[i]);
            int ed = (i % 2 == 0 ? msg_l[i] : valueAt(i0, dotsNumber));
            XYZ point;
            for(setValueAt(i0, st, &point);  valueAt(i0, point) < ed; addValueAt(i0, 1, &point)) {
                for(setValueAt(i1, 0, &point);  valueAt(i1, point) < valueAt(i1, dotsNumber); addValueAt(i1, 1, &point)) {
                    for(setValueAt(i2, 0, &point);  valueAt(i2, point) < valueAt(i2, dotsNumber); addValueAt(i2, 1, &point)) {
                        long long int uIndex = scalar(point, uSize);
                        int mult = (i % 2 == 0 ? valueAt(i0, point) : valueAt(i0, point) - ed + msg_l[i]);
                        int sendIndex = valueAt(i1, point) * srHelper[i0][0] + valueAt(i2, point) * srHelper[i0][1] + mult * srHelper[i0][2];
                        send[i][sendIndex] = u.y[uIndex];
                    }
                }
            }
        }

//        инициализация пересылок
        int cnt = 0;
        for(int i = 0; i < 2 * 3; ++i){
            if (msg_l[i] > 0){
                (i % 2 == 0) ? addValueAt(i / 2, -1, &coordinate) : addValueAt(i / 2, 1, &coordinate);
                if (convertPointToRank(range, rankMultiplier, point) == rank) {
                    //если блок отправляет данные сам себе, пересылки на самом деле не нужны
                    double *tmp = send[i];
                    send[i] = recv[i % 2 == 0 ? i + 1 : i - 1];
                    recv[i % 2 == 0 ? i + 1 : i - 1] = tmp;
                } else {
                    int rank = convertPointToRank(range, rankMultiplier, coordinate);
                     MPI_Send_init(send[i], msg_l[i] * srHelper[i / 2][2], MPI_DOUBLE, rank, i % 2, MPI_COMM_WORLD, &(request[cnt]));
                    cnt += 1;
                     MPI_Recv_init(recv[i], msg_l[i] * srHelper[i / 2][2], MPI_DOUBLE, rank, (i + 1) % 2, MPI_COMM_WORLD, &(request[cnt]));
                    cnt += 1;
                }
                (i % 2 == 0) ? addValueAt(i / 2, -1, &coordinate) : addValueAt(i / 2, 1, &coordinate);
            }
        }
//        пересылки
        MPI_Startall(cnt, request);
        MPI_Waitall(cnt, request, MPI_STATUSES_IGNORE);

        //используем гибридную версию распараллеливания
        //#pragma omp parallel for num_threads(2)

//        //считаем Лапласса и u^n+1 за одно
        for (point.x = 0; point.x < dotsNumber.x; point.x += 1) {
            for (point.y = 0; point.y < dotsNumber.y; point.y += 1) {
                for (point.z = 0; point.z < dotsNumber.z; point.z += 1) {
                    double delta = 0, d[3];
                    long long int ind;
                    bool f = false;
                    for(int i = 0; i < 3; ++i){
                        double ub = 0, uc, uf;
                        ind = scalar(point, uSize);
                        uc = u.y[ind];

                        if (valueAt(i, point) == 0) {
                            if (msg_l[2 * i] == 0) {
                                uc = 0;
                                f = true;
                            }
                            long long int recvIndex = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                            if (msg_l[2 * i] == 1) {
                                ub = recv[2 * i][recvIndex];
                            }
                            if (msg_l[2 * i] == 2){
                                //ub = recv; uc = recv;
                                ub = recv[2 * i][recvIndex];
                                uc = recv[2 * i][recvIndex];
                            }
                        } else{
                            addValueAt(i, -1, &point);
                            ind = scalar(point, uSize);
                            ub = u.y[ind];
                            addValueAt(i, 1, &point);
                        }
                        if (valueAt(i, point) == valueAt(i, dotsNumber) - 1) {
                            if (msg_l[2 * i + 1] == 0){
                                uf = -ub; uc = 0;
                                f = true;
                            }
                            long long int recvIndex = valueAt(index_inf[2 * i], point) * srHelper[i][0] + valueAt(index_inf[2 * i + 1], point) * srHelper[i][1];
                            if (msg_l[2 * i + 1] == 1) {
                                uf = recv[2 * i + 1][recvIndex];
                            }
                            if (msg_l[2 * i + 1] == 2){
                                //ub = recv; uc = recv;
                                uc = recv[2 * i + 1][recvIndex];
                                uf = recv[2 * i + 1][recvIndex];
                            }
                        } else {
                            addValueAt(i, 1, &point);
                            ind = scalar(point, uSize);
                            uf = u.y[ind];
                            addValueAt(i, -1, &point);
                        }
                        if (valueAt(i, point) == 0 && msg_l[2 * i] == 0) {
                            ub = -uf;
                        }
                        d[i] = ub - 2 * uc + uf;
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

        //перебрасываем указатели, экономим память
        double *tmp = u.z;
        u.z = u.x;
        u.x = u.y;
        u.y = tmp;
        t += tau;
    }

    syncThreads();
    executionTime = MPI_Wtime() - executionTime;

    norm = 0;
    for(point.x = 0; point.x < dotsNumber.x; point.x += 1) {
        for(point.y = 0; point.y < dotsNumber.y; point.y += 1) {
            for(point.z = 0; point.z < dotsNumber.z; point.z += 1) {
                long long int ind = scalar(point, uSize);
                FXYZ fpoint = finit(baseCoordinate.x + step.x * point.x,
                                    baseCoordinate.y + step.y * point.y,
                                    baseCoordinate.z + step.z * point.z);
                double diff = u.y[ind] - ut(fpoint, t);
                norm += diff * diff;
            }
        }
    }

    if (rank == 0) {
        double *norms = calloc(processCount, sizeof(double));
        norms[0] = norm;
        MPI_Request *r = malloc(processCount * sizeof(MPI_Request));
        for(int i = 1; i < processCount; ++i){
            MPI_Recv_init(&(norms[i]), 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(r[i - 1]));
        }
        MPI_Startall(processCount - 1, r);
        MPI_Waitall(processCount - 1, r, MPI_STATUSES_IGNORE);
        norm = 0;
        for(int i = 0; i < processCount; ++i){
            norm += norms[i];
        }
        norm = sqrt(norm);
        printf("norm over all is %f; time: %f\n", norm, executionTime);
        free(r);
    } else {
        printf("norm for process %d is %lf", rank, norm);
        MPI_Send(&norm, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    free(u.x);
    free(u.y);
    free(u.z);
    for(int i = 0; i < 3; ++i){
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

void looping() {
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < 10; i++) {
    }
}

int main(int argc, char * argv[]) {
    int processCount, rank = 10;
    double executionTime = 0;
    int gridSize;
    double gridSteps;
    AXYZ u;
    XYZ range, rankMultiplier, coordinate, dotsNumber;
    FXYZ step, baseCoordinate;
    
    L = finit(M_PI, M_PI, M_PI);
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
    convertRankToPoint(rank, range, &coordinate);
    printf("Rank is %d", rank);
    print(coordinate);
    initIteratorParams(gridSize, &step, &baseCoordinate, &dotsNumber, coordinate, range);
    
    XYZ uSize = init(1, dotsNumber.x, dotsNumber.x * dotsNumber.y);
    calculateU(&u, uSize, dotsNumber, baseCoordinate, step);
    calculate(dotsNumber, coordinate, range, uSize, u, rankMultiplier, step, baseCoordinate, rank, gridSteps, processCount, executionTime);
    
    finilize();
    return 0;
}

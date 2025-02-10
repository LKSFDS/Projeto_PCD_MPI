#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define N 2000       
#define DELTA_T 0.01
#define DELTA_X 1.0
#define D 0.1       
#define ITERATIONS 501 

void initialize(double **C, int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = (i == N / 2 && j == N / 2) ? 1.0 : 0.0;
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double start_time = MPI_Wtime();

    int rows_per_proc = N / size;
    int remainder = N % size;
    int start, end;
    double aux_final = 0.0;
    
    if (rank < remainder) {
        start = rank * (rows_per_proc + 1);
        end = start + (rows_per_proc + 1);
    } else {
        start = rank * rows_per_proc + remainder;
        end = start + rows_per_proc;
    }

    double **C = (double **)malloc(N * sizeof(double *));
    double **C_new = (double **)malloc(N * sizeof(double *));
    if (!C || !C_new) {
        printf("Erro ao alocar memória!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < N; i++) {
        C[i] = (double *)malloc(N * sizeof(double));
        C_new[i] = (double *)malloc(N * sizeof(double));
        if (!C[i] || !C_new[i]) {
            printf("Erro ao alocar memória na linha %d!\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    initialize(C, start, end);

    MPI_Request send_req[2], recv_req[2];
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        
        if (rank > 0) {
            MPI_Isend(C[start], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_req[0]);
            MPI_Irecv(C[start - 1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_req[0]);
        }
        if (rank < size - 1) {
            MPI_Isend(C[end - 1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_req[1]);
            MPI_Irecv(C[end], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_req[1]);
        }
        
        if (rank > 0) MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        if (rank < size - 1) MPI_Wait(&recv_req[1], MPI_STATUS_IGNORE);

        double diff_sum = 0.0;
        int count = 0;
        for (int i = start; i < end; i++) {
            for (int j = 1; j < N - 1; j++) {
                if (i > 0 && i < N - 1) {
                    C_new[i][j] = C[i][j] + D * DELTA_T * (
                        (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                    );
                    diff_sum += fabs(C_new[i][j] - C[i][j]);
                    count++;
                    
                    //if( iter == 500){ printf("Concentração final no centro: %f\n", C[N/2][N/2]); break;}
                }
            }
        }
        
        double global_diff_sum;
        MPI_Reduce(&diff_sum, &global_diff_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && iter % 100 == 0) {
            printf("Iteração %d, Diferença Média: %e\n", iter, global_diff_sum / (N * N));
        }

        double **temp = C;
        C = C_new;
        C_new = temp;
    }   
    
    if (rank == 0) {
        printf("Concentração final no centro: %f\n", C[N/2][N/2]);
    }

    double end_time = MPI_Wtime();
 
    if (rank == 0) {
        printf("Tempo total de execução: %f segundos\n", end_time - start_time);
    }
    
    for (int i = 0; i < N; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);
    
    MPI_Finalize();
    return 0;
}

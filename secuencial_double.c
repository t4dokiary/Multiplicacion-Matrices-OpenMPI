#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MAX 10000
#define float double
void Read_matrix (char *prompt, float *matrix, int n);
void Print_matrix (char *prompt, float *matrix, int n);
void Set_to_zero (float *matrix, int n);

int main(int argc, char* argv[]) {
	srand(time(NULL));
	int i, j, k;
	int n;
	int c = 0;
	float *matrix_A, *matrix_B, *matrix_C;
	double start, end;
	double temp;
	int my_rank, p;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (my_rank == 0) {

		printf("\t\t**********************************************\n");
		printf("\t\t*            Algoritmo Secuencial            *\n");
		printf("\t\t**********************************************\n\n");

		if (argc == 2) {
			sscanf(argv[1], "%d", &n);
		} else {
			printf("¿ Cuál es el orden de las matrices ? \n");
			scanf("%d", &n);
		}

		matrix_A = (float *) malloc (MAX*MAX*sizeof(float));
		matrix_B = (float *) malloc (MAX*MAX*sizeof(float));
		matrix_C = (float *) malloc (MAX*MAX*sizeof(float));

		Read_matrix ("Ingrese A : ", matrix_A, n);
		Print_matrix ("Se leyó A : ", matrix_A, n);

		Read_matrix ("Ingrese B : ", matrix_B, n);
		Print_matrix ("Se leyó B : ", matrix_B, n);

		Set_to_zero(matrix_C, n);

		start = MPI_Wtime();
	/*************************************************/
	/* Algoritmo secuencial para la multiplicación de matrices*/
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				for (k = 0; k < n; k++) {
					matrix_C[i + j*n] += matrix_A[i + k*n]*matrix_B[k*n + j];
				}
			}
		}
	/************************************************/
		end = MPI_Wtime();
		MPI_Finalize();

		Print_matrix ("El producto es C : ", matrix_C, n);
		printf("n : %d\nSecuencial : %f segundos.\n", n, end - start);

		return 0;
	} else {	/*Si no es el primer procesador entonces finaliza.*/
		MPI_Finalize();
		return 0;
	}
}

void Read_matrix (char *prompt, float *matrix, int n) {
	int i, j;
	printf("%s\n", prompt);
	for(i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
                long long int val=1;
                while(val<=99999999){
                    val*=rand()%100;
                }
				matrix[i*n + j] = 1000 + rand()%1000 +(float)((float)val/(float)(100000000)  ) ;
			}
	}
} /* Read_matrix */

void Print_matrix (char *prompt, float *matrix, int n)  {
	int i, j;
	printf("%s\n", prompt);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			printf("%15.5f ", matrix[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
} /* Print_matrix */

void Set_to_zero (float * matrix, int n) {
	int i, j;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			matrix[j + i*n] = 0.0;
} /* Set_to_zero */

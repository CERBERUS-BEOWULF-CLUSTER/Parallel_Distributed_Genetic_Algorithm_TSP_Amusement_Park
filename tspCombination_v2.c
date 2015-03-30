/*
 * File: globals.h
 *
 * Combination of the Global Variable, Master and Slave
 * Algorithm translated from PVM to MPI
 *
 * Copyright 1997. Giuseppe A. Sena
 *
 * COM3480: Machine Learning. Fall 1997
 *
 * Edited from PVM to MPI 2015.  Megan A. DeLaney
 *
 */

/*                                                                           */
/***************************  Include Files  *********************************/
/*                                                                           */
#include <stdio.h>
#include <malloc.h>
#include <sys/time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>                                         /* include mpi */


/* macro definitions */
#define SWAP(a,b,temp) (temp) = (a); (a) = (b); (b) = (temp);

/* global type definitions */
typedef enum { FALSE , TRUE } BOOLEAN;

/* Constants */
#define NO_EDGE       -1
#define MAX_N        150
#define MAX_FILENAME 128

/* The largest number rand will return (same as INT_MAX).  */
#ifndef RAND_MAX
#define RAND_MAX        2147483647
#endif

/* message tags */
#define     MASTER_TAG 10
#define      SLAVE_TAG 11
#define PARAMETERS_TAG 12
#define    MINIMUM_TAG 13
#define       STOP_TAG 14
#define  MIGRATION_TAG 15
#define      SYNCH_TAG 16

/* MPI parameters */
int info;                               /* value returned by send operations */
int bufid;                       /* buffer ID returned by receive operations */
#define MASTER 0
MPI_Status status;
MPI_Request request;
int nslaves;                                   /* number of slaves processes */
int my_rank;                                                   /* my task id */
int comm_sz;                                          /* number of slaves +1 */

/* Timer variables */
double start, finish;                       /* START and STOP for simulation */                                       
float  tot_time;                     /* time in seconds taken by computation */

/* GA parameters */

/* INPUT parameters */
int   p;    /* total # of hamiltonian paths to be included in the population */
int   ps;                                /* # of hamiltonian paths per slave */
int   max_gen;                                       /* max # of generations */
float r;          /* % of the paths to be replaced by Crossover at each step */
float m;                                                    /* mutation rate */
int   mut_int;     /* mutation interval: # generations between each mutation */
int   pop_mutate;              /* # of paths to mutate in current population */
float mig_rate;          /* percentage of individuals selected for migration */
int   mig_int;   /* migration interval: # generations between each migration */
int   min_threshold;                /* minimum cost accepted to stop program */

/* COMPUTED parameters */
int   pop_keep;                    /* # of paths to keep for next generation */
int   pop_cross;                          /* # of paths to use for crossover */
int   pop_migrate;                 /* # of paths to migrate to neighbor task */
int   cross_pairs;                                          /* pop_cross / 2 */
int  *Keep_Path;                        /* paths to keep for next generation */
int  *Cross_Path;          /* paths to use for crossover for next generation */
int  *son_1;                     /* one of the result paths from a crossover */
int  *son_2;                     /* one of the result paths from a crossover */
int   generation = 1;            /* indicates current generation (iteration) */
int   min;            /* index for the minimum tourlenght path in population */
int   stop = FALSE;                             /* TRUE if solution is found */

/* Graph Parameters */
typedef struct path_obj {
  float  Pr;                      /* probability of each path being selected */
  float  F;                 /* acummulated distributed function: Pr{x <= Xi} */
  int    cost;                                               /* cost of path */
  int   *path;                                        /* pointer to the path */
} PATH_OBJ;

int       n;                                   /* # of vertices in the graph */
char      fn_cost[MAX_FILENAME];          /* data file with cost information */
int       Cost[MAX_N][MAX_N];                          /* cost of every edge */
PATH_OBJ *Paths;                             /* array of paths for eachslave */
int      *Sorted_Paths;       /* array of vertices sorted in ascending order */
int      *rand_tour;

/* Parameters for timing measurement */
struct timeval tv;

/*****************************************************************************/
/************************ MASTER GLOBAL VARIABLES ****************************/
/*****************************************************************************/

int  min_cost,  slave_min_cost;
int *min_path, *slave_min_path;

/*****************************************************************************/
/************************* SLAVE GLOBAL VARIABLES ****************************/
/*****************************************************************************/

int me;                              /* indicates slave number=[0,nslaves-1] */
int master;                                            /* TID of master task */
int tmp, tmp_cost, *tmp_path;

/*****************************************************************************/
/***************************** MASTER FUNCTIONS ******************************/
/*****************************************************************************/

void allocate_memory_master()
{
  slave_min_path     = (int *)calloc(n, sizeof(int));
  min_path = (int *)calloc(n, sizeof(int));
}

/*****************************************************************************/

void recv_min_path_master()                                  
{
  int   i=1;
  int slave_rank = 0;  
  
  /* get first result */
  info = MPI_Recv(&slave_rank, 1, MPI_INT, MPI_ANY_SOURCE, MINIMUM_TAG, MPI_COMM_WORLD, &status);
  info = MPI_Recv(&slave_min_cost, 1, MPI_INT, MPI_ANY_SOURCE, MINIMUM_TAG, MPI_COMM_WORLD, &status);
  info = MPI_Recv(slave_min_path, n, MPI_INT, MPI_ANY_SOURCE, MINIMUM_TAG, MPI_COMM_WORLD, &status);
  if (generation == 1) {
    min_cost = slave_min_cost;                                   /* initializes min_cost */
    printf("MIN_COST starts as %d\n\n", min_cost);
    fflush(stdout);
  }
  /* get next results and compute minimum cost path */
  while (i < nslaves) {
    info = MPI_Recv(&slave_rank, 1, MPI_INT, MPI_ANY_SOURCE, MINIMUM_TAG, MPI_COMM_WORLD, &status);
    info = MPI_Recv(&slave_min_cost, 1, MPI_INT, MPI_ANY_SOURCE, MINIMUM_TAG, MPI_COMM_WORLD, &status);
    info = MPI_Recv(slave_min_path, n, MPI_INT, MPI_ANY_SOURCE, MINIMUM_TAG, MPI_COMM_WORLD, &status);
    
    if (slave_min_cost < min_cost) {
      min_cost = slave_min_cost;
      SWAP(min_path, slave_min_path, tmp_path);
    };
    i++;
  };
}

/*****************************************************************************/

void print_solution_master()
{
  int i;
  if (stop == TRUE)
    printf("Minimum-Cost Path found after %d generations\n\n", generation);
  else
    printf("Best Minimum-Cost Path found after %d max number generations\n\n",
	   generation-1);

  /* print minimum path found */
  printf("Minimum-Cost Path:\tCOST = %d\n\n", min_cost);
  for (i = 0; i < n; i++)
    printf("V%d->", min_path[i]);
  printf("V%d\n\n", min_path[0]);
}

/*****************************************************************************/

void send_stop_to_slaves_master()
{
  int i;
  for (i = 1; i <= nslaves; i++) {
	info = MPI_Send(&stop, 1, MPI_INT, i, STOP_TAG, MPI_COMM_WORLD);
  }
}

/*****************************************************************************/
/***************************** SLAVE FUNCTIONS *******************************/
/*****************************************************************************/

void get_parameters_slave()                                        
{
  /* set "pop_cross" and "pop_keep" */
  pop_cross   = (int)(r * (float)p);
  cross_pairs = pop_cross / 2;
  if ((pop_cross % 2) != 0)
    pop_cross++;
  pop_keep    = p - pop_cross;

  pop_mutate  = (int)(m        * (float)p);         /* set # paths to mutate */
  pop_migrate = (int)(mig_rate * (float)p);         /* set # paths to mutate */
  
  /* initialize random number generator */

  gettimeofday(&tv, NULL);
  srand((unsigned)tv.tv_usec + (unsigned)getpid());
}
/*****************************************************************************/

void alloc_memory_slave()                                          
{
  int  i;

  Paths        = (PATH_OBJ *)calloc(p, sizeof(PATH_OBJ));
  Sorted_Paths = (int *)calloc(p, sizeof(int));
  for (i = 0; i < p ; i++) {
    Paths[i].cost = 0.0;
    Paths[i].path = (int *)calloc(n, sizeof(int));
    Sorted_Paths[i] = i;
  };

  /* paths to keep for next generation and to do crossover */
  Keep_Path  = (int *)calloc(pop_keep,  sizeof(int));
  Cross_Path = (int *)calloc(pop_cross, sizeof(int));

  /* two paths resulting from a crossover */
  son_1 = (int *)calloc(n, sizeof(int));
  son_2 = (int *)calloc(n, sizeof(int));

  rand_tour = (int *)calloc(n, sizeof(int));
}

/*****************************************************************************/

void read_data_file_slave()                         
{
  int i, j, n, edge_cost;
  char fn[MAX_FILENAME] = "./";                      /* input file */
  FILE *fd;

  strcat(fn, fn_cost);
  if ((fd = fopen(fn, "r")) == NULL) {
    perror("fopen()");
    exit(1);
  };
  fscanf(fd, "%d", &n);
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      fscanf(fd, "%d", &edge_cost);
      Cost[i][j] = edge_cost;
    };
  };
  n = fclose(fd);
}

/*****************************************************************************/
/* Generate "p" hamiltonian paths at random.                                 */
/*****************************************************************************/

void init_paths_slave()                                        
{
  int i, j, v, vertex, *path;
  
  for (i = 0; i < p; i++) {
    for (v = 0; v < n; v++) rand_tour[v] = FALSE;
    path = Paths[i].path;
    for (j = 0; j < n; j++) {                /* generate hamiltonian circuit */
      vertex = (int)(((float)(n - 1) * (float)rand()) / (float)RAND_MAX);
      while (rand_tour[vertex] == TRUE)
	vertex = (vertex + 1) % n;
      path[j] = vertex;
      rand_tour[vertex] = TRUE;      
    }; /* end for j */
  }; /* end for i */

}

/*****************************************************************************/

int compute_cost_slave(int *path)
{
  int v, c = 0;

  for (v = 0; v < n ; v++)
    c += Cost[path[v]][path[(v + 1) % n]];

  return c;
}

/*****************************************************************************/

void compute_fitness_slave()                               
{
  int  i;

  for (i = 0; i < p; i++)
    Paths[i].cost = compute_cost_slave(Paths[i].path);
}

/*****************************************************************************/

void compute_probabilities_slave()                        
{
  int i;
  int tot_fit = 0;

  /* compute probability density function */
  for (i = 0; i < p; i++)
    tot_fit += Paths[i].cost;
  for (i = 0; i < p; i++)
    Paths[i].Pr = (float)Paths[i].cost / (float)tot_fit;

  /* compute acummulated distributed function */
  Paths[0].F =  Paths[0].Pr;
  for (i = 1; i < p; i++)
    Paths[i].F = Paths[i-1].F + Paths[i].Pr;
}

/*****************************************************************************/

int select_random_path_slave()                                
{
  int   i;
  float val;
  
  val = (float)rand() / (float)RAND_MAX;                         /* in [0,1] */
  for (i = 0; i < p; i++) {
    if (val <= Paths[i].F) return i;
  };
  return (p-1);
}

/*****************************************************************************/

void select_paths_to_keep_slave()                         
{
  int i, j, k;
  BOOLEAN ok;

  Keep_Path[0] = select_random_path_slave();
  for (i = 1; i < pop_keep; i++) {
    /* each path selected is different */
    ok = FALSE;
    while (!ok) {
      ok = TRUE;
      k  = select_random_path_slave();
      for (j = 0; j < i; j++) {
	if (Keep_Path[j] == k) {
	  ok = FALSE;
	  break;
	};
      }; /* end for j */
    }; /* end while */
    Keep_Path[i] = k;
  }; /* end for i */
}

/*****************************************************************************/

void select_paths_to_crossover_slave()                        
{
  int  i, j, k;
  BOOLEAN ok;

  Cross_Path[0] = select_random_path_slave();
  for (i = 1; i < pop_cross; i++) {
    /* each path selected is different */
    ok = FALSE;
    while (!ok) {
      ok = TRUE;
      k  = select_random_path_slave();
      for (j = 0; j < (i-1); j++) {
	if (Cross_Path[j] == k) {
	  ok = FALSE;
	  break;
	};
      };
    };
    Cross_Path[i] = k;
  };
}

/*****************************************************************************/

void build_son_slave(int *o1, int *p1, int *p2, int a, int b)
{
  int i, j, k, ok;

  /* copy sub-sequence of vertices from positions "a" to "b" from "p1" */
  for (i = a; i <= b; i++)
    o1[i] = p1[i];

  /* copy vertices from "p2" starting from "b+1", skiping existing vertices */
  j = b + 1;
  k = j;
  while (j != a) {
    /* check if vertex "p2[k]" is already in sub-sequence o1[a..b] */
    ok = FALSE;
    while (!ok) {
      ok = TRUE;
      for (i = a; i <= b; i++) {
	if (p2[k] == o1[i]) {
	  k  = (k + 1) % n;
	  ok = FALSE;
	  break;
	};
      }; /* end for i */
    }; /* end while */

    o1[j] = p2[k];
    k     = (k + 1) % n;
    j     = (j + 1) % n;
  };
}

/*****************************************************************************/

void crossover_slave(int mom, int dad)
{
  int *p1, *p2;
  int a, b;                               /* cut points with 0 < a < b < n-1 */

  /* generate two random cut points */
  a = (int)(((float)(n-2) * (float)rand()) / (float)RAND_MAX) + 1;
  b = (int)(((float)(n-2) * (float)rand()) / (float)RAND_MAX) + 1;
  while (a == b)
    b = (int)(((float)(n-2) * (float)rand()) / (float)RAND_MAX) + 1;
  if (a > b) {
    SWAP(a, b, tmp);
  };

  p1 = Paths[mom].path;                                        /* mom's path */
  p2 = Paths[dad].path;                                        /* dad's path */

  build_son_slave(son_1, p1, p2, a, b);
  build_son_slave(son_2, p2, p1, a, b);
} 

/*****************************************************************************/

void generate_offspring_slave()                            /* DONE */
{
  int  i;
  int  mom, dad;                                   /* Cost[mom] <= Cost[dad] */
  int  c1, c2;                                 /* Cost[son_1] <= Cost[son_2] */

  for (i = 0; i < cross_pairs; i++) {
    /* make sure Cost[mom] <= Cost[dad] is true */
    mom = Cross_Path[i];
    dad = Cross_Path[i+1];
    if (Paths[mom].cost > Paths[dad].cost) {
      SWAP(mom, dad, tmp);
    };

    crossover_slave(mom, dad);            /* must update paths "son_1" and "son_2" */

    /* make sure Cost[son_1] <= Cost[son_2] is true */
    c1 = compute_cost_slave(son_1);
    c2 = compute_cost_slave(son_2);
    if (c1 > c2) {
      SWAP(c1, c2, tmp_cost);
      SWAP(son_1, son_2, tmp_path);
    };

    /* replace parents if necessary */
    if (c2 < Paths[mom].cost) {
      SWAP(c2, Paths[mom].cost, tmp_cost);
      SWAP(son_2, Paths[mom].path, tmp_path);
    };
    if (c1 < Paths[dad].cost) {
      SWAP(c1, Paths[dad].cost, tmp_cost);
      SWAP(son_1, Paths[dad].path, tmp_path);
    };
  };
}

/*****************************************************************************/
/* mutates "pop_mutate" paths from current population. It chooses random     */
/* paths using a uniform distribution. For each one of those paths it        */
/* selects two positions at random and swaps them.                           */
/*****************************************************************************/

void mutate_paths_slave()                                       
{
  int i;
  int j;                                   /* random path to mutate: [0,p-1] */
  int k, l;                      /* random vertices to swap in path: [0,n-1] */

  for (i = 0; i < pop_mutate; i++) {
    k = (int)(((float)(n-1) * (float)rand()) / (float)RAND_MAX);
    l = (int)(((float)(n-1) * (float)rand()) / (float)RAND_MAX);
    if (k != l) {
      j = (int)(((float)(p-1) * (float)rand()) / (float)RAND_MAX);
      SWAP(Paths[j].path[k], Paths[j].path[l], tmp);
      Paths[j].cost = compute_cost_slave(Paths[j].path);
    };
  };
}

/*****************************************************************************/

void sort_paths_slave()                                      
{
  int i, j, min;

  for (i = 0; i < p-1; i++) {
    min = i;
    for (j = i+1; j < p; j++) {
      if (Paths[j].cost < Paths[min].cost) min = j;       /* ascending order */
    };
    SWAP(Sorted_Paths[i], Sorted_Paths[min], tmp);
  };
}

/*****************************************************************************/

int compute_minimum_path_slave()                              
{
  int i, min = 0;

  for (i = 1; i < p; i++) {
    if (Paths[i].cost < Paths[min].cost) min = i;
  };

  return min;
}

/*****************************************************************************/

void migrate_paths_slave() /* Isend and Ireceive require different status inputs */                               
{
  int i, k;
  MPI_Request status_isend;

  for (i = 0; i < pop_migrate; i++) {
    k    = Sorted_Paths[i];                                   /* path number */
    info = MPI_Isend(&Paths[k].cost, 1, MPI_INT, (my_rank + 1) % nslaves, MIGRATION_TAG, MPI_COMM_WORLD, &status_isend); /* path cost */
    info = MPI_Isend(Paths[k].path, n, MPI_INT, (my_rank + 1) % nslaves, MIGRATION_TAG, MPI_COMM_WORLD, &status_isend); /* path */
    };
      
  /* receive "pop_migrate" best paths from neighbor task (me - 1) */
  for (i = p - pop_migrate; i < p; i++) {
    k    = Sorted_Paths[i];                                   /* path number */
    info = MPI_Irecv(&Paths[k].cost, 1, MPI_INT, (my_rank - 1) % nslaves, MIGRATION_TAG, MPI_COMM_WORLD, &request); /* path cost */
    info = MPI_Irecv(Paths[k].path, n, MPI_INT, (my_rank - 1) % nslaves, MIGRATION_TAG, MPI_COMM_WORLD, &request); /* path */
  };
}

/*****************************************************************************/

void send_minimum_path_slave(int min)               
{
  info = MPI_Send(&my_rank, 1, MPI_INT, MASTER, MINIMUM_TAG, MPI_COMM_WORLD);
  info = MPI_Send(&Paths[min].cost, 1, MPI_INT, MASTER, MINIMUM_TAG, MPI_COMM_WORLD);
  info = MPI_Send(Paths[min].path, n, MPI_INT, MASTER, MINIMUM_TAG, MPI_COMM_WORLD);
}

/*****************************************************************************/

void recv_confirmation_slave()                          
{
  info = MPI_Recv(&stop, 1, MPI_INT, MASTER, STOP_TAG, MPI_COMM_WORLD, &status);
}

/*****************************************************************************/
/******************************* MAIN PROGRAM ********************************/
/*****************************************************************************/

/* Main from MASTER */
int main(int argc, char *argv[])
{
  /* Enroll in MPI */
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Barrier(MPI_COMM_WORLD);
  nslaves = comm_sz - 1;
  
  if (argc != 11) {                /* get input parameters from command line */
    printf("\n\nError: master_tsp_ga <cost_filenane> <n> <p> <r> ");
    printf("<m> <mut_int> <mig_rate> <mig_int> <max_gen> <min_threshold>");
    exit(1);
  } else {
    strcpy(fn_cost, argv[1]);
    n             = atoi(argv[2]);
    p             = atoi(argv[3]);
    r             = atof(argv[4]);
    m             = atof(argv[5]);
    mut_int       = atoi(argv[6]);
    mig_rate      = atof(argv[7]);
    mig_int       = atoi(argv[8]);
    max_gen       = atoi(argv[9]);
    min_threshold = atoi(argv[10]);
  };
  if (my_rank == 0) {
  printf("Input Parameters: <cost_filename> = %s\t <n> = %d\t <p> = %d\t <r> = %f\t <m> = %f\t <mut_int> = %d\t <mig_rate> = %f\t <mig_int> = %d\t <max_gen> = %d\t <min_threshold> = %d\t <nslaves> = %d\t\n\n", fn_cost, n, p, r, m, mut_int, mig_rate, mig_int, max_gen, min_threshold, nslaves);
fflush(stdout);
  };
  if (my_rank != 0) {
    MPI_Barrier(MPI_COMM_WORLD);                                            /* Blocks slaves until master ready */
  }
  /* adjust "p" to be a multiple of "nslaves" */
  if (p < nslaves) {
    exit(1);
  } else if ((p % nslaves) != 0)
    p = nslaves * ((p / nslaves) + 1);
  ps = p / nslaves;
  
  /*                                                                         */
  /*********************  Begin PGA MASTER Program  **************************/
  /*                                                                         */
  
  /***************************************************************************/
  /**************************** IF MASTER ************************************/
  /***************************************************************************/
  if (my_rank == 0) {  

  printf("\nProblem Parameters:\n");
  printf("Number of Slaves: nslaves=%d\tTotal Population: p=%d\n",
	 nslaves, p);
  printf("Fraction of Population replaced by Crossover: r=%6.3f\n", r);
  printf("Mutation  Rate: m       =%6.3f\tMutation  Interval: mut_int=%d\n",
	 m, mut_int);
  printf("Migration Rate: mig_rate=%6.3f\tMigration Interval: mig_int=%d\n",
	 mig_rate, mig_int);
  printf("Maximum Number of Generations: max_gen=%d\tThreshold=%d\n",
	 max_gen, min_threshold);
  printf("Input Data File = %s\n\n", fn_cost);
  fflush(stdout);
  
  allocate_memory_master();
  
  MPI_Barrier(MPI_COMM_WORLD);                                        /* Unblocks slaves */
  start = MPI_Wtime();                                                   /* start time */

  /* MASTER LOOP */
  while ((generation <= max_gen) && (stop == FALSE)) {
    recv_min_path_master();
    stop = ((min_cost <= min_threshold) ? TRUE : FALSE);
    send_stop_to_slaves_master();
    generation++;  
  };

  finish = MPI_Wtime();                                                    /* stop time */
  printf("Proc %d > Elapsed time = %e seconds\n", my_rank, finish - start);
  print_solution_master();
  }
  else { 
  /***************************************************************************/
  /****************************** ELSE SLAVE *********************************/
  /***************************************************************************/
  get_parameters_slave();         /* receive data from master - KEEP FOR CROSS POP */
  alloc_memory_slave();                 /* allocate memory for all data structures */
  read_data_file_slave();
  init_paths_slave();                  /* generate "p" hamiltonian paths at random */

  /* PGA (Parallel Genetic Algorithm) */
  while ((generation <= max_gen) && (stop == FALSE)) {
    compute_fitness_slave();
    compute_probabilities_slave();
    select_paths_to_keep_slave();
    select_paths_to_crossover_slave();
    generate_offspring_slave();
    
    /* mutate paths if necessary, based on mutation interval */
    if ((pop_mutate > 0) && ((generation % mut_int) == 0))
      mutate_paths_slave();
    /* migrate paths to successor slave in the ring */
    if ((nslaves > 1) && ((generation % mig_int) == 0)) {
      sort_paths_slave();
      migrate_paths_slave();
      min = Sorted_Paths[0];
    } else
      min = compute_minimum_path_slave();
    send_minimum_path_slave(min);
    recv_confirmation_slave();
    generation++;
  };
  }
  MPI_Finalize();                /* Program Finished exit MPI before stopping */
  return 0;
}


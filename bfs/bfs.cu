/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "infalloc.h"

#include <sys/time.h>
#include <time.h>
double diff(struct timespec start, struct timespec end)
{
   struct timespec temp;
   if ((end.tv_nsec-start.tv_nsec)<0) {
      temp.tv_sec = end.tv_sec-start.tv_sec-1;
      temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
   } else {
      temp.tv_sec = end.tv_sec-start.tv_sec;
      temp.tv_nsec = end.tv_nsec-start.tv_nsec;
   }
   if(temp.tv_sec)
      return (double)(temp.tv_sec*1000000000+temp.tv_nsec);
   else
      return (double)temp.tv_nsec;

}



#define MAX_THREADS_PER_BLOCK 512

int no_of_nodes;
int edge_list_size;
FILE *fp;

#include "node.h"
#include "kernel.cu"
#include "kernel2.cu"


Node *graph_nodes;
bool *graph_mask;
bool *updating_graph_mask;
bool *graph_visited;
int *graph_edges;
int *m_cost;
bool *over;


void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

    char *input_f;
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}


   struct timespec start_time, end_time;
   double TotalMallocManagedTime = 0.0;
   clock_gettime(CLOCK_REALTIME, &start_time);

	// allocate host memory
   inf_init();
   graph_nodes=(Node *)infalloc(sizeof(Node)*no_of_nodes,DEFAULT);
	graph_mask=(bool *)infalloc(sizeof(bool)*no_of_nodes,DEFAULT);
	updating_graph_mask=(bool *)infalloc(sizeof(bool)*no_of_nodes,DEFAULT);
	graph_visited=(bool *)infalloc(sizeof(bool)*no_of_nodes,DEFAULT);


   clock_gettime(CLOCK_REALTIME, &end_time);
   TotalMallocManagedTime+=diff(start_time,end_time)/1000000.0;

	int start, edgeno;

   struct timespec start_time1, end_time1;
   double indirectManagedOverhead = 0.0;
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
      clock_gettime(CLOCK_REALTIME, &start_time1);
		graph_nodes[i].starting = start;
		graph_nodes[i].no_of_edges = edgeno;
		graph_mask[i]=false;
		updating_graph_mask[i]=false;
		graph_visited[i]=false;
      clock_gettime(CLOCK_REALTIME, &end_time1);
      indirectManagedOverhead+=diff(start_time1,end_time1)/1000000.0;
	}


	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	graph_mask[source]=true;
	graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
   clock_gettime(CLOCK_REALTIME, &start_time);
	graph_edges=(int *)infalloc(sizeof(int)*edge_list_size,DEFAULT);
   clock_gettime(CLOCK_REALTIME, &end_time);
   TotalMallocManagedTime+=diff(start_time,end_time)/1000000.0;
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
      clock_gettime(CLOCK_REALTIME, &start_time1);
		graph_edges[i] = id;
      clock_gettime(CLOCK_REALTIME, &end_time1);
      indirectManagedOverhead+=diff(start_time1,end_time1)/1000000.0;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

   clock_gettime(CLOCK_REALTIME, &start_time);
	// allocate mem for the result on host side
	m_cost=(int *)infalloc(sizeof(int)*no_of_nodes,DEFAULT);
   clock_gettime(CLOCK_REALTIME, &end_time);
   TotalMallocManagedTime+=diff(start_time,end_time)/1000000.0;
	
   clock_gettime(CLOCK_REALTIME, &start_time1);
   for(int i=0;i<no_of_nodes;i++)
		m_cost[i]=-1;
	m_cost[source]=0;
   clock_gettime(CLOCK_REALTIME, &end_time1);
   indirectManagedOverhead+=diff(start_time1,end_time1)/1000000.0;

   clock_gettime(CLOCK_REALTIME, &start_time);   
	//make a bool to check if the execution is over
	over=(bool *)infalloc(sizeof(bool),DEFAULT);
   clock_gettime(CLOCK_REALTIME, &end_time);
   TotalMallocManagedTime+=diff(start_time,end_time)/1000000.0;

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
   cudaEvent_t begin, stop;
   cudaEventCreate(&begin);
   cudaEventCreate(&stop);
   cudaEventRecord(begin, 0);

   clock_gettime(CLOCK_REALTIME, &start_time);
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
      *over = false;
		Kernel<<< grid, threads, 0 >>>( graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, m_cost, no_of_nodes);
      cudaDeviceSynchronize();
		// check if kernel execution generated and error
		

		Kernel2<<< grid, threads, 0 >>>( graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes);
      cudaDeviceSynchronize();
		// check if kernel execution generated and error
		
		k++;
	}
	while(*over);
   clock_gettime(CLOCK_REALTIME, &end_time);

   cudaDeviceSynchronize(); // this is really important! I have been debugging for while before relizing this is necessary!
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, begin, stop);

   printf("[CUDA EVENTS] Cuda kernel Processing time:  %f (ms)\n", elapsedTime);
   printf("[CPU TIMESPEC] Cuda kernel Processing time: %f (ms)\n", diff(start_time,end_time)/1000000.0);
   printf("Total cudaMallocManaged time:     %f (ms)\n", TotalMallocManagedTime);
   printf("Indirect Managed memory overhead: %f (ms)\n", indirectManagedOverhead);
   printf("Kernel Executed %d times\n",k);

	//Store the result into a file
	//FILE *fpo = fopen("result.txt","w");
	//for(int i=0;i<no_of_nodes;i++)
	//	fprintf(fpo,"%d) cost:%d\n",i,m_cost[i]);
	//fclose(fpo);
	//printf("Result stored in result.txt\n");


	// cleanup memory
   inf_shutdown();
}

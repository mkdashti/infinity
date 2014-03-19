
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h> 

__global__ void infinity(int *index_array,
                         int *data_array, 
                         unsigned long array_size, 
                         int nthreads, 
                         unsigned long portion_size,
                         int do_shuffle, 
                         int read_bench) {

   int idx = blockDim.x*blockIdx.x +threadIdx.x;
   int temp,garbage = 0;
   int i;
   int startof_portion, endof_portion;

   if(idx < nthreads) {

      if(do_shuffle) {
         startof_portion = idx*portion_size;
         endof_portion = startof_portion + portion_size;

         if(read_bench) {
            for(i=startof_portion; i<endof_portion ; i++)
               temp = data_array[index_array[i]];
            if(temp < 7)
               garbage++;
         }
         else {
            for(i=startof_portion; i<endof_portion ; i++)
               data_array[index_array[i]] = 1;
         }
      }
      else {
         if(read_bench) {
            for(i=0; i<=(array_size-portion_size); i=i+portion_size)
               temp = data_array[index_array[idx+i]];
            if(temp < 7)
               garbage++;
         }
         else {
            for(i=0; i<=(array_size-portion_size); i=i+portion_size)
               data_array[index_array[idx+i]] = 1;
         }
      }
   }
}

/* Arrange the N elements of ARRAY in random order.
 *    Only effective if N is much smaller than RAND_MAX;
 *       if this may not be the case, use a better random
 *          number generator. */
void shuffle(int *array, size_t n)
{
   if (n > 1) 
   {
      size_t i;
      for (i = 0; i < n - 1; i++) 
      {
         size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
         int t = array[j];
         array[j] = array[i];
         array[i] = t;
      }
   }
}

void usage(void)
{
   printf("Usage:\n");
   printf(" -u <unified memory> (0/1)\n");
   printf(" -t <t> (number of threads)\n");
   printf(" -s <shuffle> (1/0)\n");
   printf(" -m <number of integer entries in data array in Mega>\n");
   printf(" -h print usage\n");
   printf(" -d debug\n");
   exit (EXIT_FAILURE);
}

int main(int argc, char **argv) {

   int i;
   int nthreads = 1024;

   /* Data and buffers */
   int *message, *data_array, *read_array;
   int *d_message, *d_data_array, *d_read_array;
   //unsigned long  message_size = 1024*1024*128;  //larger than 200 sometimes makes some fuctions fail.
   unsigned long  message_size = 1024;  //larger than 200 sometimes makes some fuctions fail.
   //unsigned long checksum = 0;

   int opt;
   int do_shuffle=0;
   int debug=0;
   int read_bench=0;
   int unified=0;

   while ((opt = getopt(argc, argv, "u:t:s:m:d:r:h:")) != -1) {
      switch (opt) {
         case 'u':
            unified = atoi(optarg);
            break;
         case 't':
            nthreads = atoi(optarg);
            break;
         case 's':
            do_shuffle = atoi(optarg);
            break;
         case 'm':
            message_size = atoi(optarg)*1024*1024;
            break;
         case 'd':
            debug = atoi(optarg);
            break;
         case 'h':
            usage();
            break;
         case 'r':
            read_bench = atoi(optarg);
            break;
         default: /* '?' */
            usage();
      }
   }

   if((message_size%nthreads)!=0) {
      perror("invalid buffer size or number of threads \n");
      exit(1);
   }  

   unsigned long portion_size;
   if (do_shuffle) 
      portion_size = message_size/nthreads;
   else
      portion_size = nthreads;

   if(unified)
   {
      cudaMallocManaged((void**)&message, sizeof(int)*message_size);
      cudaMallocManaged((void**)&data_array, sizeof(int)*message_size);
      cudaMallocManaged((void**)&read_array, sizeof(int)*message_size);
   }
   else
   {

      message = (int*)malloc(sizeof(int)*message_size);
      data_array = (int*)malloc(sizeof(int)*message_size);
      read_array = (int*)malloc(sizeof(int)*message_size);

      cudaMalloc(&d_message, sizeof(int)*message_size);
      cudaMalloc(&d_data_array, sizeof(int)*message_size);
      cudaMalloc(&d_read_array, sizeof(int)*message_size);
   }

   for(i=0; i< message_size; i++) {
      message[i]=i;
   }

   if(do_shuffle) {
      //printf("shuffling the index array ...\n");
      shuffle(message,message_size);
   }

   /*
   checksum = 0;
   for(i=0; i< message_size; i++) {
      checksum += message[i];
   }
   printf("Checksum after shuffle = %lu\n",checksum);
*/
   for(i=0; i< message_size; i++) {
      data_array[i]=i;
   }

   if(!unified)
   {
      cudaMemcpy(d_message, message, sizeof(int)*message_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_data_array, data_array, sizeof(int)*message_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_read_array, read_array, sizeof(int)*message_size, cudaMemcpyHostToDevice);
   }

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   if(unified) {
      infinity<<< nthreads/64, 64 >>>(message,data_array,message_size,nthreads,portion_size,do_shuffle,read_bench);
      cudaDeviceSynchronize();
   }
   else
      infinity<<< nthreads/64, 64 >>>(d_message,d_data_array,message_size,nthreads,portion_size,do_shuffle,read_bench);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);

   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, start, stop);

   if(!unified)
   {
      cudaMemcpy(message, d_message, sizeof(int)*message_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(data_array, d_data_array, sizeof(int)*message_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(read_array, d_read_array, sizeof(int)*message_size, cudaMemcpyDeviceToHost);

   }
   //Confirm that the resulting data array is correct (all values have become = 1)
   if(debug) {
      for(i=0; i<message_size; i++)
         printf("%d ", data_array[i]);   
   }

   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   if(unified)
   {
      cudaFree(message);
      cudaFree(data_array);
      cudaFree(read_array);
   }
   else
   {
      free(message);
      free(data_array);
      free(read_array);
      cudaFree(d_message);
      cudaFree(d_data_array);
      cudaFree(d_read_array);
   }
   cudaDeviceReset();

   printf("Cuda kernel Processing time: %f (ms)\n", elapsedTime);

   return 0;
}

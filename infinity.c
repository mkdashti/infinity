#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "infinity.cl"
#define KERNEL_FUNC "infinity"
#define _GNU_SOURCE 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h> 

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <sys/time.h>
#include <time.h>

int *global_data_array, *global_index_array;
unsigned long global_portion_size; 

unsigned long diff(struct timespec start, struct timespec end)
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
      return (temp.tv_sec*1000000000+temp.tv_nsec);
   else
      return temp.tv_nsec;

}

static pid_t gettid(void) {                                                                      
   return syscall(__NR_gettid);
}

void set_affinity(int tid, int core_id) {
   cpu_set_t mask;
   CPU_ZERO(&mask);
   CPU_SET(core_id, &mask);

   int r = sched_setaffinity(tid, sizeof(mask), &mask);                                                                  
   if (r < 0) {
      fprintf(stderr, "couldn't set affinity for %d\n", core_id);
      exit(1);
   }
}


typedef struct {                                                                                                        
   int id;
} parm;

void *work( void *arg ) {
   parm *p=(parm *)arg;
   int tid = gettid();                                                                                                   
   int i;
  // printf("Assigning thread %d (tid = %d) to core %d\n", p->id, tid, (p->id)%get_nprocs());
   set_affinity(tid, (p->id)%get_nprocs());

   int startof_portion = (p->id)*global_portion_size;
   int endof_portion = startof_portion + global_portion_size;
   for(i=startof_portion; i<endof_portion ; i++)
      global_data_array[global_index_array[i]] = 1;

   return 0;
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

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}
void usage(void)
{
   printf("Usage:\n");
   printf(" -g <g> (run on GPU '1' on CPU '0'\n");
   printf(" -t <t> (number of threads)\n");
   printf(" -s <shuffle> (1/0)\n");
   printf(" -m <number of integer entries in data array in Mega>\n");
   printf(" -h print usage\n");
   printf(" -d debug\n");
   exit (EXIT_FAILURE);
}

int main(int argc, char **argv) {

   /* OpenCL data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int err;
   int i;
   cl_ulong max_buffer_size;
   size_t global_work_size = 1024;
   int nthreads = global_work_size;
   cl_event ev_enqueue_map_buffer, 
            ev_enqueue_kernel, 
            ev_enqueue_read_buffer,
            ev_enqueue_unmap_buffer;

   cl_ulong starttime[4], endtime[4];
   struct timespec start_time, end_time;

   /* Data and buffers */
   cl_mem msg_buffer, data_buffer;
   int *message, *data_array;
   //unsigned long  message_size = 1024*1024*128;  //larger than 200 sometimes makes some fuctions fail.
   unsigned long  message_size = 1024;  //larger than 200 sometimes makes some fuctions fail.
   //unsigned long checksum = 0;

   pthread_t *threads;
   pthread_attr_t attr; 
   parm *p;
   int j,opt;
   int gpu_run=0;
   int do_shuffle=0;
   int debug=0;

   while ((opt = getopt(argc, argv, "g:t:s:m:d:")) != -1) {
      switch (opt) {
         case 'g':
            gpu_run = atoi(optarg);
            break;
         case 't':
            nthreads = atoi(optarg);
            global_work_size = nthreads;
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

   message = (int *)malloc(sizeof(int)*message_size);
   if(message == NULL) {
      perror("malloc failed \n");
      exit(1);   
   }
   data_array = (int *)malloc(sizeof(int)*message_size);
   if(data_array == NULL) {
      perror("malloc failed \n");
      exit(1);   
   }

   // cheking that shuffling the array doesn't remove/repeat elements by naively summing
   // all the elements of the array and considering the sum as the checksum...ok
   //
   for(i=0; i< message_size; i++) {
      message[i]=i;
      //checksum += message[i];
   }
   //printf("Inititial checksum before shuffle = %lu\n",checksum);

   if(do_shuffle) {
      printf("shuffling the index array ...\n");
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
   
   /////////////// pthread version /////////////////
   //
   // The same exact version as the opencl except
   // we have fewer threads accessing the 128MB array
   //
   // /////////////////////////////////////////////

   global_data_array = data_array;
   global_index_array = message;
   global_portion_size = portion_size;

   if(!gpu_run) {
      threads=(pthread_t *)malloc(nthreads * sizeof(pthread_t));
      if(threads == NULL) {
         printf("ERROR malloc failed to create CPU threads\n");
         exit(1);
      }
      pthread_attr_init(&attr);

      p=(parm *)malloc(nthreads * sizeof(parm));

      clock_gettime(CLOCK_REALTIME, &start_time);
      for (j=0; j<nthreads; j++)
      {
         p[j].id=j;
         if(pthread_create(&threads[j], &attr, work, (void *)(p+j))!=0)
         {
            printf("ERROR creating threads\n");
            exit(1);
         }
      }

      for (j=0; j<nthreads; j++)
      {
         if(pthread_join(threads[j],NULL)!=0) {
            printf("ERROR in joing threads\n");
            exit(1);
         }
      }
      clock_gettime(CLOCK_REALTIME, &end_time);

      pthread_attr_destroy(&attr);
      free(p);

      printf("Time to execute on CPU = %lu ns = %lu ms\n",
            diff(start_time,end_time),
            diff(start_time,end_time)/1000000L);
      //Confirm that the resulting data array is correct (all values have become = 1)
      if(debug) {
         for(i=0; i<message_size; i++)
            printf("%d ", data_array[i]);   
      }

   }

   if(gpu_run) {
      /* Create a device and context */
      device = create_device();
      context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
      if(err < 0) {
         perror("Couldn't create a context");
         exit(1);   
      }

      /* Build a program and create a kernel */
      program = build_program(context, device, PROGRAM_FILE);
      kernel = clCreateKernel(program, KERNEL_FUNC, &err);
      if(err < 0) {
         perror("Couldn't create a kernel");
         exit(1);   
      };

      /* Create a command queue */
      queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
      if(err < 0) {
         perror("Couldn't create a command queue");
         exit(1);   
      };


      /* Create a buffer to hold the output data */
      //msg_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(msg), msg, &err);
      //
      //allocate buffer on host-accessible device memory and copy the content from
      //the host buffer to the device buffer
      msg_buffer = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
            sizeof(int)*message_size,
            message,
            &err);

      if(err < 0) {
         perror("Couldn't create a buffer");
         exit(1);   
      };

      message =  (int *)clEnqueueMapBuffer(queue,
            msg_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
            0,
            sizeof(int)*message_size,
            0,
            NULL,
            &ev_enqueue_map_buffer,
            &err);

      if(err < 0) {
         perror("Map output buffer failed");
         exit(1);   
      }

      /////////////////
      //same as above
      //
      /////////////////
      data_buffer = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
            sizeof(int)*message_size,
            data_array,
            &err);

      if(err < 0) {
         perror("Couldn't create a buffer");
         exit(1);   
      };

      data_array =  (int *)clEnqueueMapBuffer(queue,
            data_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
            0,
            sizeof(int)*message_size,
            0,
            NULL,
            NULL,
            &err);

      if(err < 0) {
         perror("Map output buffer failed");
         exit(1);   
      }

      ///////////////////////////
      ///////////////////////////
      //////////////////////////

      /* Create kernel argument */
      err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &msg_buffer);
      if(err < 0) {
         perror("Couldn't set a kernel argument");
         exit(1);   
      };
      err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &data_buffer);
      if(err < 0) {
         perror("Couldn't set a kernel argument");
         exit(1);   
      };
      err = clSetKernelArg(kernel, 2, sizeof(unsigned long), &message_size);
      if(err < 0) {
         perror("Couldn't set a kernel argument");
         exit(1);   
      };

      err = clSetKernelArg(kernel, 3, sizeof(int), &nthreads);
      if(err < 0) {
         perror("Couldn't set a kernel argument");
         exit(1);   
      };
      err = clSetKernelArg(kernel, 4, sizeof(unsigned long), &portion_size);
      if(err < 0) {
         perror("Couldn't set a kernel argument");
         exit(1);   
      };
      err = clSetKernelArg(kernel, 5, sizeof(int), &do_shuffle);
      if(err < 0) {
         perror("Couldn't set a kernel argument");
         exit(1);   
      };



      /* Enqueue kernel */
      /*err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
        if(err < 0) {
        perror("Couldn't enqueue the kernel");
        exit(1);   
        }*/
      err = clEnqueueNDRangeKernel(queue,
            kernel,
            1,
            NULL,
            &global_work_size,
            NULL,
            0,
            NULL,
            &ev_enqueue_kernel);

      if(err < 0) {
         perror("Couldn't enqueue the kernel");
         exit(1);   
      }

      /* Read and print the result */
      err = clEnqueueReadBuffer(queue,
            data_buffer,
            CL_TRUE,
            0,
            sizeof(int)*message_size,data_array,
            0,
            NULL,
            &ev_enqueue_read_buffer);

      if(err < 0) {
         perror("Couldn't read the output buffer");
         exit(1);   
      }


      // checking the maximum buffer size allocation
      err = clGetDeviceInfo(device,
            CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            sizeof(cl_ulong),
            &max_buffer_size,
            NULL);
      if(err < 0) {
         perror("Unmap buffer failed");
         exit(1);   
      }
      printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE %lu\n", max_buffer_size);   

      //Confirm that the resulting data array is correct (all values have become = 1)
      if(debug) {
      for(i=0; i<message_size; i++)
         printf("%d ", data_array[i]);   
      }



      /* Deallocate resources */
      err = clEnqueueUnmapMemObject(queue, msg_buffer, message, 0, NULL, &ev_enqueue_unmap_buffer);
      if(err < 0) {
         perror("Unmap buffer failed");
         exit(1);   
      }
      err = clEnqueueUnmapMemObject(queue, data_buffer, data_array, 0, NULL, NULL);
      if(err < 0) {
         perror("Unmap buffer failed");
         exit(1);   
      }


      clFinish(queue);

      clGetEventProfilingInfo(ev_enqueue_map_buffer,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&starttime[0],NULL);
      clGetEventProfilingInfo(ev_enqueue_map_buffer,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endtime[0],NULL);
      clGetEventProfilingInfo(ev_enqueue_read_buffer,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&starttime[1],NULL);
      clGetEventProfilingInfo(ev_enqueue_read_buffer,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endtime[1],NULL);
      clGetEventProfilingInfo(ev_enqueue_kernel,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&starttime[2],NULL);
      clGetEventProfilingInfo(ev_enqueue_kernel,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endtime[2],NULL);
      clGetEventProfilingInfo(ev_enqueue_unmap_buffer,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&starttime[3],NULL);
      clGetEventProfilingInfo(ev_enqueue_unmap_buffer,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endtime[3],NULL);

      printf("time to enqueue map buffer = %lu ms\n", (endtime[0]-starttime[0])/1000000);
      printf("time to enqueue read buffer = %lu ms\n", (endtime[1]-starttime[1])/1000000);
      printf("time to enqueue kernel = %lu ns = %lu ms\n", (endtime[2]-starttime[2]), (endtime[2]-starttime[2])/1000000);
      printf("time to enqueue unmap buffer = %lu ms\n", (endtime[3]-starttime[3])/1000000);

      clReleaseEvent(ev_enqueue_map_buffer);
      clReleaseEvent(ev_enqueue_read_buffer);
      clReleaseEvent(ev_enqueue_unmap_buffer);
      clReleaseEvent(ev_enqueue_kernel);

      clReleaseMemObject(msg_buffer);
      clReleaseKernel(kernel);
      clReleaseCommandQueue(queue);
      clReleaseProgram(program);
      clReleaseContext(context);
   }

     return 0;
}

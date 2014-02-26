__kernel void infinity(__global int *index_array,
                                __global int *data_array, 
                                unsigned long array_size, 
                                int nthreads, 
                                unsigned long portion_size) {

size_t idx = get_global_id(0);
if(idx < nthreads) {
   int i;
   int startof_portion = idx*portion_size;
   int endof_portion = startof_portion + portion_size;
   for(i=startof_portion; i<endof_portion ; i++)
      data_array[index_array[i]] = 1;
  }

}


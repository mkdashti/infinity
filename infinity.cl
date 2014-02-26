__kernel void infinity(__global int *index_array,
                                __global int *data_array, 
                                unsigned long array_size, 
                                int nthreads, 
                                unsigned long portion_size,
                                int do_shuffle) {

size_t idx = get_global_id(0);


if(idx < nthreads) {

if(do_shuffle) {
   int i;
   int startof_portion = idx*portion_size;
   int endof_portion = startof_portion + portion_size;
   for(i=startof_portion; i<endof_portion ; i++)
      data_array[index_array[i]] = 1;
      }
else {
   int i;
   for(i=0; i<=(array_size-portion_size); i=i+portion_size)
   data_array[index_array[idx+i]] = 1;
   }
 }
}

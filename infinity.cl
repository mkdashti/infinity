__kernel void infinity(__global int *index_array,
                                __global int *data_array, 
                                unsigned long array_size, 
                                int nthreads, 
                                unsigned long portion_size,
                                int do_shuffle, 
                                int read_bench) {

size_t idx = get_global_id(0);
int temp,garbage = 0;
int i;
int startof_portion, endof_portion;
int last_portion = array_size-portion_size;

if(idx < nthreads) {

if(do_shuffle) {
   startof_portion = idx*portion_size;
   endof_portion = startof_portion + portion_size;

   if(read_bench) {
   for(i=startof_portion; i<endof_portion ; i++)
      temp = data_array[index_array[i]];
      if(temp > -1)
         garbage++;
      }
   else {
 for(i=startof_portion; i<endof_portion ; i++)
      data_array[index_array[i]] = 1;
      }
   }
else {
   if(read_bench) {
      for(i=0; i<=last_portion; i=i+portion_size)
         temp = data_array[index_array[idx+i]];
         if(temp > -1)
            garbage++;
      }
   else {
      for(i=0; i<=last_portion; i=i+portion_size)
         data_array[index_array[idx+i]] = 1;
      }
  }
 }
}

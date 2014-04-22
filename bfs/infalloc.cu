#include <assert.h>
#include "infalloc.h"
#include "node.h"

void *managed_memory;
int track=0;
extern int no_of_nodes;
extern int edge_list_size;

void *infalloc(int size, enum allocType alloc)
{
   void *return_addr=NULL;
   switch(alloc) {
      case DEFAULT: {
                      return_addr = (void *)&managed_memory[track];
                      track+=size;
                      break;
                   }
      default: {}
   }

   return return_addr; 
}
void infree(void *p)
{
}
void inf_init(void)
{
   assert(
         (cudaMallocManaged((void **)&managed_memory,
                            sizeof(Node)*no_of_nodes+
                            3*sizeof(bool)*no_of_nodes+
                            sizeof(int)*edge_list_size+
                            sizeof(int)*no_of_nodes+
                            sizeof(bool)
                            ))==cudaSuccess);
}

void inf_shutdown(void)
{
   cudaFree(managed_memory);

}

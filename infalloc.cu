#include <assert.h>
#include "btreetypes.h"
#include "infalloc.h"

extern int order;
node *nodes,*node_parents,*node_nexts;
void **pointers;
node **node_pointers;
int *keys;
record *records;

int nodes_track=0,node_parents_track=0,node_nexts_track=0,pointers_track=0,node_pointers_track=0,keys_track=0,records_track=0;


void *infalloc(int size, enum allocType alloc)
{
   void *return_addr=NULL;
   switch(alloc) {
      case NODE: {
                    return_addr = (void *)&nodes[nodes_track];
                    nodes_track+=size;
                    break;
                 }
      case NODE_PARENT: {
                           return_addr = (void *)&node_parents[node_parents_track];
                           node_parents_track+=size;
                           break;
                        }
      case NODE_NEXT: {
                         return_addr = (void *)&node_nexts[node_nexts_track];
                         node_nexts_track+=size;                    
                         break;
                      }
      case POINTER: {
                       return_addr = (void *)&pointers[pointers_track];
                       pointers_track+=size;
                       break;
                    }
      case NODE_POINTER: {
                       return_addr = (void *)&node_pointers[node_pointers_track];
                       node_pointers_track+=size;
                       break;
                    }

      case KEY: {
                   return_addr = (void *)&keys[keys_track];
                   keys_track+=size;
                   break;
                }
      case RECORD: {
                      return_addr = (void *)&records[records_track];
                      records_track+=size;
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
   assert((cudaMallocManaged((void **)&nodes,4*1024*1024*sizeof(node)))==cudaSuccess);
   assert((cudaMallocManaged((void **)&node_parents,4*1024*1024*sizeof(node)))==cudaSuccess);
   assert((cudaMallocManaged((void **)&node_nexts,4*1024*1024*sizeof(node)))==cudaSuccess);
   assert((cudaMallocManaged((void **)&pointers,4*order*1024*1024*sizeof(void*)))==cudaSuccess);
   assert((cudaMallocManaged((void **)&node_pointers,4*(order+1)*1024*1024*sizeof(node*)))==cudaSuccess);
   assert((cudaMallocManaged((void **)&keys,16*order*1024*1024*sizeof(int)))==cudaSuccess);
   assert((cudaMallocManaged((void **)&records,16*1024*1024*sizeof(record)))==cudaSuccess);


}

void inf_shutdown(void)
{
   cudaFree(nodes);
   cudaFree(node_parents);
   cudaFree(node_nexts);
   cudaFree(pointers);
   cudaFree(keys);
   cudaFree(records);

}

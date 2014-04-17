#ifndef INFALLOC_H_
#define INFALLOC_H_

enum allocType { NODE, NODE_PARENT, NODE_NEXT, POINTER, NODE_POINTER, KEY, RECORD };
extern void *infalloc(int size,enum allocType alloc);
extern void infree(void *p);
extern void inf_init(void);
extern void inf_shutdown(void);

#endif

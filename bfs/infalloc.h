#ifndef INFALLOC_H_
#define INFALLOC_H_

enum allocType { DEFAULT };
extern void *infalloc(int size,enum allocType alloc);
extern void infree(void *p);
extern void inf_init(void);
extern void inf_shutdown(void);

#endif

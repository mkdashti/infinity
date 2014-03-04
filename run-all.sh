#!/bin/bash


date >> output.txt

for MEMORY in 8 16 32 64 128
do
   for THREAD in 1 2 4 8 16
   do
      #for SHUFFLE in 0 1
      for SHUFFLE in 0
      do 
         for ITR in 0 1 2 3 4
         do
            echo "CPU $THREAD $SHUFFLE $MEMORY `./infinity -t $THREAD -s $SHUFFLE -m $MEMORY`" >> output.txt
         done
      done
   done
done

#for MEMORY in 8 16 32 64 128
#do
#   for THREAD in 128 512 1024 4096 8192 16384
#   do
#      for SHUFFLE in 0 1
#      do
#         for ITR in 0 1 2 3 4
#         do
#            echo "GPU $THREAD $SHUFFLE $MEMORY `./infinity -g 1 -t $THREAD -s $SHUFFLE -m $MEMORY`" >> output.txt
#         done
#      done
#   done
#done

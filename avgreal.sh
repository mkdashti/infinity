#!/bin/bash

grep "$1" | sed -e s/"$1"// -e 's/\t//' | awk '{sum+=$1} END {print sum/NR}'

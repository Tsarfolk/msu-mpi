#!/bin/bash -x
rm -f main core.*
mpixlc_r -qsmp=omp main.c -o main

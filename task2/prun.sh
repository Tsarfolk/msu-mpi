#!/bin/bash -x 
mpisubmit.pl -p $1 -w 00:30  main  $2 $3

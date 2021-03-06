#!/bin/bash

if [ $1 = "512" ] 
then
	mpisubmit.bg -n $1 -w 00:05:00 -m smp -e "OMP_NUM_THREADS=2" --stdout output.log --stderr error.log main $2 $3
elif [ $1 = "256" ]
then
	mpisubmit.bg -n $1 -w 00:10:00 -m smp -e "OMP_NUM_THREADS=2" --stdout output.log --stderr error.log main $2 $3
else
	mpisubmit.bg -n $1 -w 00:15:00 -m smp -e "OMP_NUM_THREADS=2" --stdout output.log --stderr error.log main $2 $3
fi
#mpisubmit.bg -n $1 -w 00:15:00 -m smp -e "OMP_NUM_THREADS=2" main $2 $3 $4

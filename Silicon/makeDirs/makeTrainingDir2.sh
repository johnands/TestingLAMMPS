#!/bin/bash 

DATE=`date +%d.%m-%H.%M.%S`
dir="Data/TrainingData/2atoms/$DATE"
mkdir $dir

copy1="log.lammps $dir"
move2="Data/TrainingData/neighbours.txt $dir"
copy2="../../lammps/src/pair_mysw.cpp $dir"
cp $copy1
mv $move2
cp $copy2

#!/bin/bash 

DATE=`date +%d.%m-%H.%M.%S`
dir="Data/TrainingData/4atoms/$DATE"
mkdir $dir

copy1="log.lammps $dir"
copy2="../../lammps/src/pair_myvashishta.cpp $dir"
move1="Data/TrainingData/neighbours0.txt $dir"
move2="Data/TrainingData/neighbours1.txt $dir"

cp $copy1
cp $copy2

mv $move1
mv $move2

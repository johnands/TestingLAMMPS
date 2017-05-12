#!/bin/bash 

DATE=`date +%d.%m-%H.%M.%S`
dir="Data/Forces/$DATE"
mkdir $dir

copy1="log.lammps $dir"
move2="Data/Forces/forces.txt $dir"
copy2="../../lammps/src/pair_myvashishta.cpp $dir"
cp $copy1
mv $move2
cp $copy2

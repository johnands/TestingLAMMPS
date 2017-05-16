#!/bin/bash 

DATE=`date +%d.%m-%H.%M.%S`
dir="Data/Si/Thermo/$DATE"
mkdir $dir

copy1="log.lammps $dir"
move2="Data/Si/Thermo/thermo.txt $dir"
copy2="../../lammps/src/pair_nn_angular2.cpp $dir"
cp $copy1
mv $move2
cp $copy2




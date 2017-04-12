#!/bin/bash 

DATE=`date +%d.%m-%H.%M.%S`
dir="Data/Thermo/$DATE"
mkdir $dir

move1="log.lammps $dir"
#move2="Data/Thermo/pairForces.txt $dir"
#move3="Data/Thermo/tripletForces.txt $dir"
move2="Data/Thermo/thermo.txt $dir"
copy1="../../lammps/src/pair_nn_angular2.cpp $dir"
mv $move1
mv $move2
cp $copy1




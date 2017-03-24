Trying to validate the implementation to obtain forces in lammps by writing out
all neighbour coordinates of one random atom (not on the edge) at one time step, thus
creating a data set consisting of only one input vector and one energy. 

A NN is then trained, and the error in the energy quickly drops to zero. 
The same neighbour vector is then loaded in my lammps implementation of NN potential to 
see if we get the correct energy. 

We do get the correct energy, but not correct forces. Is that because my implementation is incorrect
or because the error in the forces is not necessarily good? 

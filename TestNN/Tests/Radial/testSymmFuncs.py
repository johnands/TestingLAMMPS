"""
Test whether I have implemented the symmetry functions correctly in C++
Compare evaluation of a specific Rij here and in C++
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandParentDir = os.path.dirname(parentdir)
gGrandParentDir = os.path.dirname(grandParentDir)
ggGrandGrandParentDir = os.path.dirname(gGrandParentDir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, grandParentDir)
sys.path.insert(2, gGrandParentDir)
sys.path.insert(3, ggGrandGrandParentDir)

import TensorFlow.DataGeneration.symmetries as symmetries
import numpy as np

with open("../../LAMMPS_test/TestNN/testSymmFuncs.dat", 'r') as infile:
    Rij = []
    for line in infile:
       Rij.append(float(line))
       
Rij = np.array(Rij)

# parameters
cutoffs = [8.5125]
widths = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.3, 0.7]
centers = [0.0, 3.1, 4.5, 5.2, 5.9, 6.8, 7.8]

evaluation = []
for width in widths:
    for cutoff in cutoffs:
        for center in centers:                                           
            evaluation.append(symmetryFunctions.G2(Rij, cutoff, width, center))
            
print evaluation
        

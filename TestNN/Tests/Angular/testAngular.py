"""
Test whether I have implemented the symmetry functions correctly in pair_nn_angular.cpp
I have written 1 neighbour list, together with input vector after symmetry transformations
and energy after network evaluation. 
Compare evaluation of a specific Rij here and in C++
Used TrainingData/20.02-16.36.28 to run the lammps test
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

x = []; y = []; z = []; r = []
with open("testAngular.dat", 'r') as infile:
    for line in infile:
    	xi = []; yi = []; zi = []; ri = []
        words = line.split()
        N = (len(words)) / 4
        for i in xrange(N):
            xi.append(float(words[4*i]))
            yi.append(float(words[4*i+1]))
            zi.append(float(words[4*i+2]))
            ri.append(float(words[4*i+3]))
        x.append(xi)
        y.append(yi)
        z.append(zi)
        r.append(ri)
        break
 
x = np.array(x)
y = np.array(y)
z = np.array(z)
r = np.array(r)


# make rjk and theta
#for j in xrange(len(x)): 
#    for k in xrange(

# parameters
# parameters G2
widthG2 = [0.001, 0.01, 0.1]
cutoffG2 = [4.0]
centerG2 = [0.0, 3.0]

# parameters G4
widthG4 = [0.001, 0.01]      
cutoffG4 = [4.0]
thetaRangeG4 = [1, 2, 4] 
inversionG4 = [1.0, -1.0]

# make nested list of all symetry function parameters
parameters = []
for width in widthG2:
    for cutoff in cutoffG2:
        for center in centerG2:           
            parameters.append([width, cutoff, center])
         
for width in widthG4:   
    for cutoff in cutoffG4:
        for zeta in thetaRangeG4:
            for inversion in inversionG4:
                parameters.append([width, cutoff, zeta, inversion])

E = [[-1.17132]]
        
# apply symmetry transformation to input data and generate output data
inputData, outputData = symmetries.applyThreeBodySymmetry(x, y, z, r, parameters, E=E)
print inputData

        

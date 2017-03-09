"""
Test whether I have implemented the symmetry functions correctly in pair_nn_angular.cpp
I have written 1 neighbour list, together with input vector after symmetry transformations
and energy after network evaluation. Written NOT for the first time step.
Compare evaluation of a specific Rij here and in C++
Used the NN in TrainingData/03.03-17.57.14. 
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
with open("inputVector.txt", 'r') as infile:
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
r2 = r**2
print x
print y
print z
print r

# make nested list of all symetry function parameters
# parameters from Behler
parameters = []    

# type1
center = 0.0
cutoff = 6.0
for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
    parameters.append([eta, cutoff, center])

# type2
zeta = 1.0
inversion = 1.0
eta = 0.01
for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
    parameters.append([eta, cutoff, zeta, inversion])
    
# type 3
cutoff = 6.0
eta = 4.0
for center in [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
    parameters.append([eta, cutoff, center])
    
    
eta = 0.01

# type 4
zeta = 1.0
inversion = -1.0    
for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
    parameters.append([eta, cutoff, zeta, inversion])
    
# type 5 and 6
zeta = 2.0
for inversion in [1.0, -1.0]:
    for cutoff in [6.0, 5.0, 4.0, 3.0]:
        parameters.append([eta, cutoff, zeta, inversion])
    
# type 7 and 8
zeta = 4.0
for inversion in [1.0, -1.0]:
    for cutoff in [6.0, 5.0, 4.0, 3.0]:
        parameters.append([eta, cutoff, zeta, inversion])

# type 9 and 10
zeta = 16.0
for inversion in [1.0, -1.0]:
    for cutoff in [6.0, 4.0]:
        parameters.append([eta, cutoff, zeta, inversion])  

E = [[-4.18582]]
        
# apply symmetry transformation to input data and generate output data
inputData, outputData = symmetries.applyThreeBodySymmetry(x, y, z, r2, parameters, E=E)
print inputData

        

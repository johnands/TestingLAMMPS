"""
Test whether I have implemented the symmetry functions correctly in pair_nn_angular.cpp
I have written 1 neighbour list, together with input vector after symmetry transformations
and energy after network evaluation. Written NOT for the first time step.
Compare evaluation of a specific Rij here and in C++
Used the NN in TrainingData/03.03-17.57.14. 

Also test derivatives of symmetry functions. I have implemented them in python 
and checked if I get the same answer when differentiating in c++ and python. I do, 
but this is not a real check...
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
with open("inputVector2.txt", 'r') as infile:
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

testDerivatives = True

# test derivatives of symmetryfunctions
if testDerivatives:
    
    derivatives = np.zeros((1,len(parameters)))
    
    xi = x[0]
    yi = y[0]
    zi = z[0]
    ri = r[0]
    numberOfNeighbours = len(xi)
    
    for j in xrange(numberOfNeighbours):
                      
        # atom j
        rij = ri[j]
        xij = xi[j]; yij = yi[j]; zij = zi[j]
        
        # all k != i,j OR I > J ???
        k = np.arange(len(ri[:])) > j  
        rik = ri[k] 
        xik = xi[k]; yik = yi[k]; zik = zi[k]

        # compute cos(theta_ijk) and rjk
        cosTheta = (xij*xik + yij*yik + zij*zik) / (rij*rik) 
        
        # floating-point error can yield an argument outside of arccos range
        if not (np.abs(cosTheta) <= 1).all():
            for l, arg in enumerate(cosTheta):
                if arg < -1:
                    cosTheta[l] = -1
                    print "Warning: %.14f has been replaced by %d" % (arg, cosTheta[l])
                if arg > 1:
                    cosTheta[l] = 1
                    print "Warning: %.14f has been replaced by %d" % (arg, cosTheta[l])
        
        rjk = np.sqrt( rij**2 + rik**2 - 2*rij*rik*cosTheta )
        
        # find value of each symmetry function for this triplet
        symmFuncNumber = 0
        for s in parameters:
            if len(s) == 3:
                drij = symmetries.dG2dr(rij, s[0], s[1], s[2])
            else:
                print "rij: ", rij
                print "rik: ", rik
                print "rik: ", rjk
                print "cosTheta: ", cosTheta
                print "xij: ", xij
                print "yij: ", yij
                print "zij: ", zij
                print "xik: ", xik
                print "yik: ", yik
                print "zik: ", zik
                print s[0]
                print s[1]
                print s[2]
                print s[3]
                dij, dik = symmetries.dG4dr(rij, rik, rjk, cosTheta, \
                                            xij, xik, yij, yik, zij, zik, \
                                            s[0], s[1], s[2], s[3])
                print "G4: ", dij[0], dij[1], dij[2], dik[0], dik[1], dik[2]
                exit(1)
            symmFuncNumber += 1
            
    print derivatives


	
else:       
	# apply symmetry transformation to input data and generate output data
	inputData, outputData = symmetries.applyThreeBodySymmetry(x, y, z, r2, parameters, E=E)
	print inputData















        

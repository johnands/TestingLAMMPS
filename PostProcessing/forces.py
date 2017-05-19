"""
Analyze forces dumped with LAMMPS: 
*   Plot forces of empirical potential that is reproduced and 
    NN forces as function of time step to compare
    (both real simulations and pseudo-simulations)
*   Calculate various error estimates
*   Check that sum of forces is zero



"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandParentDir = os.path.dirname(parentdir)
gGrandParentDir = os.path.dirname(grandParentDir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, grandParentDir)
sys.path.insert(2, gGrandParentDir)

import TensorFlow.DataGeneration.readers as readers

sumOfForcesFlag = True
plotDistFlag = False
plotCoordsFlag = False
    
def readForceFile(filename):
   
    with open(filename, 'r') as infile:
        
        force = False  
        timeStep = False                    
        forces = []
        timeSteps = []
        
        # read number of atoms and first time step
        infile.readline()
        timeSteps.append(int(infile.readline()))
        
        infile.readline()
        numberOfAtoms = int(infile.readline())
        print "Number of atoms: ", numberOfAtoms
      
        i = 0
        for line in infile:
            words = line.split()
            
            if words[-1] == 'TIMESTEP':
                timeStep = True
                continue
                
            if words[-1] == 'fz':
                force = True
                continue
            
            if timeStep:
                timeSteps.append(int(words[0]))
                timeStep = False
                continue
                
            if force:
                i += 1
                forcei = []
                forcei.append(float(words[1]))
                forcei.append(float(words[2]))
                forcei.append(float(words[3]))
                forces.append(forcei)
                if i == numberOfAtoms:
                    i = 0
                    force = False 
           
    return np.array(timeSteps), np.array(forces), numberOfAtoms

    

# read force files
dirNameNN = '../TestNN/Data/SiO2/Forces/InitialConfigReplicate4/'
dirNameSW = '../Quartz/Data/Forces/InitialConfigReplicate4/'

# write out README files
print "Content of SW folder:"
os.system('cat ' + dirNameSW + 'README.txt')
print "Content of NN folder:"
os.system('cat ' + dirNameNN + 'README.txt')

# read all forces in force file
timeStepsNN, forcesNN, numberOfAtomsNN = readForceFile(dirNameNN + 'forces.txt')
timeStepsSW, forcesSW, numberOfAtomsSW = readForceFile(dirNameSW + 'forces.txt')
assert(numberOfAtomsNN == numberOfAtomsSW)
numberOfAtoms = numberOfAtomsNN

# check that time step arrays of SW and NN are equal
if not np.array_equal(timeStepsNN, timeStepsSW):
    print 'Forces for NN and SW must be sampled for the same time steps'
    exit(1)
else:
    timeSteps = timeStepsNN    

nTimeSteps = len(timeSteps)
print "Number of time steps: ", nTimeSteps
    
# access components
FxNN = forcesNN[:,0]
FyNN = forcesNN[:,1]
FzNN = forcesNN[:,2]

FxSW = forcesSW[:,0]
FySW = forcesSW[:,1]
FzSW = forcesSW[:,2]

# check that sum of forces is zero
if sumOfForcesFlag:
    sumsNN = np.zeros((nTimeSteps, 3))
    sumsSW = np.zeros((nTimeSteps, 3))
    for i in xrange(nTimeSteps):
        timeStep = numberOfAtoms*i
        sumsNN[i][0] = np.sum(FxNN[timeStep:timeStep+numberOfAtoms])
        sumsNN[i][1] = np.sum(FyNN[timeStep:timeStep+numberOfAtoms])
        sumsNN[i][2] = np.sum(FzNN[timeStep:timeStep+numberOfAtoms])
        sumsSW[i][0] = np.sum(FxSW[timeStep:timeStep+numberOfAtoms])
        sumsSW[i][1] = np.sum(FySW[timeStep:timeStep+numberOfAtoms])
        sumsSW[i][2] = np.sum(FzSW[timeStep:timeStep+numberOfAtoms])
        
    if (sumsNN > 1e-4).any():
        print "Sum of forces is not zero"
        
    print "Max sum-of-forces NN: ", np.max(sumsNN)
    print "Max sum-of-forces SW: ", np.max(sumsSW)
    
# choose one atom to plot forces and calculate errors
chosenAtom = 0
print "Chosen atom: ", chosenAtom

FxNN = FxNN[np.arange(chosenAtom, len(FxNN), numberOfAtoms)]
FyNN = FyNN[np.arange(chosenAtom, len(FyNN), numberOfAtoms)]
FzNN = FzNN[np.arange(chosenAtom, len(FzNN), numberOfAtoms)]
FxSW = FxSW[np.arange(chosenAtom, len(FxSW), numberOfAtoms)]
FySW = FySW[np.arange(chosenAtom, len(FySW), numberOfAtoms)]
FzSW = FzSW[np.arange(chosenAtom, len(FzSW), numberOfAtoms)]

Fnn = np.sqrt(FxNN**2 + FyNN**2 + FzNN**2)
Fsw = np.sqrt(FxSW**2 + FySW**2 + FzSW**2)

xError = FxNN - FxSW
yError = FyNN - FySW
zError = FzNN - FzSW
absError = Fnn - Fsw

# plot error vs |F|
errorVsForce = []
forceValues = np.linspace(0, np.max(Fnn), 6)
for i in xrange(len(forceValues)-1):
    interval = np.where(Fnn < forceValues[i+1])[0]
    errorVsForce.append(np.std(absError[interval]))
    #errorVsForce.append(np.sum(absError[interval]**2)/len(interval))
plt.figure()
plt.plot(forceValues[1:], errorVsForce)
plt.xlabel('Absolute force NN')
plt.ylabel('Error')
plt.show()

# output
print "Average error Fx: ", np.average(xError)
print "Average error Fy: ", np.average(yError)
print "Average error Fz: ", np.average(zError)
print "Average error |F|: ", np.average(absError)
print
print "RMSE Fx: ", np.sqrt( np.average(xError**2) )
print "RMSE Fy: ", np.sqrt( np.average(yError**2) )
print "RMSE Fz: ", np.sqrt( np.average(zError**2) )
print "RMSE |F|: ", np.sqrt( np.average(absError**2) )
print 
print "Std. dev. error Fx: ", np.std(xError)
print "Std. dev. error Fy: ", np.std(yError)
print "Std. dev. error Fz: ", np.std(zError)
print "Std. dev. error |F|: ", np.std(absError)

# plot NN and SW forces plus the errors
plt.subplot(4,2,1)
plt.plot(timeStepsNN, FxNN, 'b-', timeStepsSW, FxSW, 'g-')
plt.legend([r'$F_x^{NN}$', r'$F_x^{SW}$'])

plt.subplot(4,2,3)
plt.plot(timeStepsNN, FyNN, 'b-', timeStepsSW, FySW, 'g-')
plt.legend([r'$F_y^{NN}$', r'$F_y^{SW}$'])   

plt.subplot(4,2,5)        
plt.plot(timeStepsNN, FzNN, 'b-', timeStepsSW, FzSW, 'g-')
plt.legend([r'$F_z^{NN}$', r'$F_z^{SW}$'])

plt.subplot(4,2,7)       
plt.plot(timeStepsNN, Fnn, 'b-', timeStepsSW, Fsw, 'g-')
plt.xlabel('Timestep')
plt.legend([r'$|F|^{NN}$', r'$|F|^{SW}$'])

plt.subplot(4,2,2)
plt.plot(timeStepsNN, xError)
plt.ylabel(r'$\Delta F_x$')

plt.subplot(4,2,4)
plt.plot(timeStepsNN, yError)
plt.ylabel(r'$\Delta F_y$')     

plt.subplot(4,2,6)        
plt.plot(timeStepsNN, zError)
plt.ylabel(r'$\Delta F_z$')

plt.subplot(4,2,8)       
plt.plot(timeStepsNN, absError)
plt.xlabel('Timestep')
plt.ylabel(r'$\Delta F$')

plt.show()

# plot histogram of forces
plt.hist(Fsw)
plt.xlabel('|F|')
plt.ylabel('Number of forces')
plt.show()


# calculate various properties of the chemical environment of the 
# chosen atom as a function of time step
# read neighbour file
if plotDistFlag or plotCoordsFlag:
    distSW = '../Quartz/Data/TrainingData/Bulk/InitConfigReplicate4/neighbours.txt'
    x, y, z, r, _ = readers.readNeighbourData(distSW)

    rAverage = np.zeros(len(r))
    rMax = np.zeros(len(r))
    rMin = np.zeros(len(r))
    rStd = np.zeros(len(r))
    coordNumber = np.zeros(len(r))
    for i in xrange(len(r)):
        ri = np.sqrt(np.array(r[i]))
        rAverage[i] = np.average(ri)
        rMax[i] = np.max(ri)
        rMin[i] = np.min(ri)
        rStd[i] = np.std(ri)
        coordNumber[i] = len(ri)
    
if plotDistFlag:    
    # plots of environment variables
    plt.subplot(4,1,1)
    plt.plot(rAverage)
    plt.legend(['rAverage'])
    plt.subplot(4,1,2)
    plt.plot(rMax)
    plt.legend(['rMax'])
    plt.subplot(4,1,3)
    plt.plot(rMin)
    plt.legend(['rMin'])
    plt.subplot(4,1,4)
    plt.plot(rStd)
    plt.legend(['rStd'])
    plt.show()  
    #plt.savefig('tmp/dist.pdf')
    
if plotCoordsFlag:
    plt.figure()
    plt.plot(timeStepsNN[0::10], coordNumber)
    plt.xlabel('Time step')
    plt.ylabel('Coordination number')
    plt.show()
    
    # correlations with coordination number
    coordUnique = np.unique(coordNumber)
    print "Coordination numbers: ", coordUnique
    plt.figure()
    plt.hist(coordNumber, bins=4)
    plt.legend(['Coordination numbers'])

    # find error as function of coordination number
    errorVsCoords = []
    for coord in coordUnique:
        indicies = np.where(coordNumber == coord)[0]
        errorVsCoords.append(np.std(absError[indicies]))
        
    plt.figure()
    plt.plot(coordUnique, errorVsCoords) 
    plt.xlabel('Coordination number')
    plt.ylabel('Error')
    plt.show()




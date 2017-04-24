import numpy as np
import matplotlib.pyplot as plt
import os
    
    
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
                forcei = []
                forcei.append(float(words[1]))
                forcei.append(float(words[2]))
                forcei.append(float(words[3]))
                forces.append(forcei)
                if int(words[0]) == numberOfAtoms:
                    force = False
           
    return np.array(timeSteps), np.array(forces), numberOfAtoms
    
    
def readNeighbourFile(filename):
    
    with open(filename, 'r') as infile:
        
        x = []; y = []; z = []
        r = [];
        for line in infile:
            words = line.split()
            N = len(words) / 4
            xi = []; yi = []; zi = [];
            ri = [];
            for i in xrange(N):
                xi.append(float(words[4*i]))
                yi.append(float(words[4*i+1]))
                zi.append(float(words[4*i+2]))
                ri.append(float(words[4*i+3]))
                
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
                      
    return x, y, z, r
    
    
def readDisplacementFile(filename):
    
    with open(filename, 'r') as infile:
        
        atoms = False  
        timeStep = False         
        dx = []; dy = []; dz = []
        dr = []
        timeSteps = []
        
        # read number of atoms and first time step
        infile.readline()
        timeSteps.append(int(infile.readline()))
        
        infile.readline()
        numberOfAtoms = int(infile.readline())
        print "Number of atoms: ", numberOfAtoms
        
        chosenAtom = numberOfAtoms/2
        print "Chosen atom: ", chosenAtom
         
        for line in infile:
            words = line.split()
            
            if words[-1] == 'TIMESTEP':
                timeStep = True
                continue
                
            if words[-1] == 'c_displacement[4]':
                atoms = True
                continue
            
            if timeStep:
                timeSteps.append(int(words[0]))
                timeStep = False
                continue
                
            if atoms:
                if int(words[0]) == chosenAtom:
                    dx.append(float(words[1]))
                    dy.append(float(words[2]))
                    dz.append(float(words[3]))
                    dr.append(float(words[4]))
                    atoms = False
                    
    return dx, dy, dz, dr
    

# read force files
dirNameNN = '../TestNN/Data/Forces/21.04-19.36.16/'
dirNameSW = '../Silicon/Data/Forces/20.04-14.12.41/'

# read displacement file
dx, dy, dz, dr = readDisplacementFile('../Silicon/tmp/diffusion.txt')

"""plt.subplot(4,1,1)
plt.plot(dx)
plt.subplot(4,1,2)
plt.plot(dy)
plt.subplot(4,1,3)
plt.plot(dz)
plt.subplot(4,1,4)
plt.plot(dr)
plt.show()"""

# read neighbour file
distSW = '../Silicon/Data/20.04-14.03.17/neighbours.txt'
x, y, z, r = readNeighbourFile(distSW)

# write out README files
print "Content of SW folder:"
os.system('cat ' + dirNameSW + 'README.txt')
print "Content of NN folder:"
os.system('cat ' + dirNameNN + 'README.txt')

readAllForces = True

if readAllForces:
    timeStepsNN, forcesNN, numberOfAtomsNN = readForceFile(dirNameNN + 'forces.txt')
    timeStepsSW, forcesSW, numberOfAtomsSW = readForceFile(dirNameSW + 'forces.txt')
    assert(numberOfAtomsNN == numberOfAtomsSW)
    numberOfAtoms = numberOfAtomsNN
else:
    timeStepsNN, forcesNN = readForceFile(dirNameNN + 'forces.txt', 1)
    timeStepsSW, forcesSW = readForceFile(dirNameSW + 'forces.txt', 1)

if not ( len(timeStepsNN) == len(timeStepsSW) and timeStepsNN.all() == timeStepsSW.all() ):
    print 'Forces for NN and SW must be sampled for the same time steps'
    exit(1)
else:
    timeSteps = timeStepsNN    

nTimeSteps = len(timeSteps)
print "Number of time steps: ", nTimeSteps
    
FxNN = forcesNN[:,0]
FyNN = forcesNN[:,1]
FzNN = forcesNN[:,2]

FxSW = forcesSW[:,0]
FySW = forcesSW[:,1]
FzSW = forcesSW[:,2]

# check that sum of forces is zero
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
    
chosenAtom = 1
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

# calculate various properties of the chemical environment of the 
# chosen atom as a function of time step
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
    
"""plt.figure()
plt.plot(timeStepsNN, coordNumber)
plt.legend('Coordination number')
plt.axis([0, 1000, 0, 3])
plt.show()"""

### investigate for correlations ###

# correlations between std. dev. of distances and force error
bottoms = np.where(rStd < 0.3)[0]
tops = np.where(rStd > 0.3)[0]

"""# displacement
errorVsDisplace = []
displaceValues = np.linspace(0, np.max(dx), 6)
for i in xrange(len(displaceValues)-1):
    interval = np.where(dx < displaceValues[i+1])[0]
    errorVsDisplace.append(np.std(xError[interval]))
    #errorVsDisplace.append( np.sqrt(np.mean(np.abs(xError[interval] - xError.mean())**2)) / len()) 
plt.plot(displaceValues[1:], errorVsDisplace)
plt.show()"""
    


# correlations with coordination number
coordUnique = np.unique(coordNumber)
print "Coordination numbers: ", coordUnique
plt.figure()
plt.hist(coordNumber, bins=4)

"""# find error as function of coordination number
coordsVsR = []
for coord in coordUnique:
    indicies = np.where(coordNumber == coord)[0]
    coordsVsR.append(np.std(absError[indicies]))
    
plt.figure()
plt.plot(coordUnique, coordsVsR) 
plt.show()"""

"""xBottoms = xError[bottoms]
xTops = xError[tops]
yBottoms = yError[bottoms]
yTops = yError[tops]
zBottoms = zError[bottoms]
zTops = zError[tops]
absBottoms = absError[bottoms]
absTops = absError[tops]"""
    
# plots of environment variables
"""plt.subplot(4,1,1)
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
#plt.show()  
#plt.savefig('tmp/dist.pdf')"""


"""print "bottoms x: ", np.std(xBottoms)
print "tops x: ", np.std(xTops)
print "bottoms y: ", np.std(yBottoms)
print "tops y: ", np.std(yTops)
print "bottoms z: ", np.std(zBottoms)
print "tops z: ", np.std(zTops)
print "bottoms abs: ", np.std(absBottoms)
print "tops abs: ", np.std(absTops)"""

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
plt.plot(timeStepsNN, xError)#, timeStepsNN, dx, 'y-')
plt.ylabel(r'$\Delta F_x$')

plt.subplot(4,2,4)
plt.plot(timeStepsNN, yError)#, timeStepsNN, dy, 'y-')
plt.ylabel(r'$\Delta F_y$')     

plt.subplot(4,2,6)        
plt.plot(timeStepsNN, zError)#, timeStepsNN, dz, 'y-')
plt.ylabel(r'$\Delta F_z$')

plt.subplot(4,2,8)       
plt.plot(timeStepsNN, absError)#, 'b-', timeStepsNN, dr, 'y-')
plt.xlabel('Timestep')
plt.ylabel(r'$\Delta F$')

plt.show()


import numpy as np
import matplotlib.pyplot as plt

def readFile(filename):
    
    with open(filename, 'r') as infile:
        
        forces = []
        for line in infile:
            fi = []
            words = line.split()
            fi.append(float(words[1]))
            fi.append(float(words[2]))
            fi.append(float(words[3]))
            forces.append(fi)
            
        forces = np.array(forces)
    
    return forces
    
forcesNN = readFile('Box/forcesNNStart.txt')
forcesSW = readFile('Box/forcesSW.txt')

nAtoms = 1000

# choose timestep and force component
timeStep = 0
timeStep /= 2
xNN = forcesNN[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 0]
xSW = forcesSW[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 0]
yNN = forcesNN[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 1]
ySW = forcesSW[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 1]
zNN = forcesNN[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 2]
zSW = forcesSW[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 2]

# get rid of outliers
"""xNN = xNN[np.where(xNN < 2)[0]]
xSW = xSW[np.where(xNN < 2)[0]]
yNN = yNN[np.where(yNN < 2)[0]]
ySW = ySW[np.where(yNN < 2)[0]]
zNN = zNN[np.where(zNN < 2)[0]]
zSW = zSW[np.where(zNN < 2)[0]]"""

# find absolute difference
diffx = xNN - xSW
diffy = yNN - ySW
diffz = zNN - zSW

# plot all components
plt.subplot(3,1,1)
plt.plot(diffx)
plt.subplot(3,1,2)
plt.plot(diffy)
plt.subplot(3,1,3)
plt.plot(diffz)
plt.show()

plt.figure()

plt.subplot(4,1,1)
plt.plot(xNN)
plt.subplot(4,1,2)
plt.plot(yNN)
plt.subplot(4,1,3)
plt.plot(zNN)
plt.subplot(4,1,4)
plt.plot(np.sqrt(xNN**2 + yNN**2 + zNN**2))
plt.show()

plt.figure()

plt.subplot(4,1,1)
plt.plot(xSW)
plt.subplot(4,1,2)
plt.plot(ySW)
plt.subplot(4,1,3)
plt.plot(zSW)
plt.subplot(4,1,4)
plt.plot(np.sqrt(xSW**2 + ySW**2 + zSW**2))
plt.show()


# averages
aveDiffx = np.sum(diffx)/nAtoms
aveDiffy = np.sum(diffy)/nAtoms
aveDiffz = np.sum(diffz)/nAtoms
print "average NN-SW x:", aveDiffx
print "average NN-SW y:", aveDiffy
print "average NN-SW z:", aveDiffz

aveDiffx = np.sum(xNN)/nAtoms
aveDiffy = np.sum(yNN)/nAtoms
aveDiffz = np.sum(zNN)/nAtoms
print "average NN x:", aveDiffx
print "average NN y:", aveDiffy
print "average NN z:", aveDiffz

aveDiffx = np.sum(xSW)/nAtoms
aveDiffy = np.sum(ySW)/nAtoms
aveDiffz = np.sum(zSW)/nAtoms
print "average SW x:", aveDiffx
print "average SW y:", aveDiffy
print "average SW z:", aveDiffz

# extremes
maxNNx = np.max(xNN)
maxNNy = np.max(yNN)
maxNNz = np.max(zNN)

maxSWx = np.max(xSW)
maxSWy = np.max(ySW)
maxSWz = np.max(zSW)

print "max NN x: ", maxNNx
print "max NN y: ", maxNNy
print "max NN z: ", maxNNz
print "max SW x: ", maxSWx
print "max SW y: ", maxSWy
print "max SW z: ", maxSWz





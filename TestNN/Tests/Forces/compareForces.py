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
    
forcesNN = readFile('forces.txt')
forcesSW = readFile('forcesSW.txt')

nAtoms = 1000

# choose timestep and force component
timeStep = 70
timeStep /= 10
xNN = forcesNN[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 0]
xSW = forcesSW[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 0]
yNN = forcesNN[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 1]
ySW = forcesSW[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 1]
zNN = forcesNN[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 2]
zSW = forcesSW[timeStep*nAtoms+1:(timeStep+1)*nAtoms, 2]

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

# averages
aveDiffx = np.sum(diffx)/nAtoms
aveDiffy = np.sum(diffy)/nAtoms
aveDiffz = np.sum(diffz)/nAtoms
print aveDiffx
print aveDiffy
print aveDiffz


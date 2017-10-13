import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

def readDisplFile(filename):
    """
    Read file dumped from LAMMPS with displace_atom
    id dx dy dz dr
    """
   
    with open(filename, 'r') as infile:
        
        displ = False  
        timeStep = False                    
        displacement = []
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
                
            if len(words) > 6:
                displ = True
                continue
            
            if timeStep:
                timeSteps.append(int(words[0]))
                timeStep = False
                continue
                
            if displ:
                i += 1
                displacement.append(float(words[4]))
                if i == numberOfAtoms:
                    i = 0
                    displ = False 
           
    return np.array(timeSteps), np.array(displacement), numberOfAtoms
    
   
timeSteps, displacement, numberOfAtoms = readDisplFile(filename)

# find mean square displacement for each time step
aveDisplacement = np.zeros(len(timeSteps))
for timeStep in xrange(len(timeSteps)):
    for atom in xrange(numberOfAtoms):
        aveDisplacement[timeStep] += displacement[timeStep*numberOfAtoms+atom]**2
        
# normalize
aveDisplacement /= numberOfAtoms

# plot to find diffusion constant
plt.plot(timeSteps, aveDisplacement)
plt.show()
    
    
    
    
    
    
    
    


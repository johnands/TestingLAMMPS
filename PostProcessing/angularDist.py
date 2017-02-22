# plot angular distribution of Si Stillinger-Weber
# file "distances.txt" is neighbour coordinates of one random
# for each time step

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

with open('../Silicon/Data/22.02-15.13.55/distances.txt', 'r') as inFile:
  
    # read neighbour lists
    x = []; y = []; z = []
    r = [];
    for line in inFile:
        words = line.split()
        N = (len(words)) / 4
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

nTimeSteps = len(x)

# compute angles
angles = []
for step in xrange(nTimeSteps):

    # convert to arrays
    xi = np.array(x[step][:])
    yi = np.array(y[step][:])
    zi = np.array(z[step][:])
    ri = np.sqrt(np.array(r[step][:]))

    nNeighbours = len(x[step])
    angleStep = []

    # loop over triplets
    for j in xrange(nNeighbours-1):

        # atom j
        rij = ri[j]
        xij = xi[j]; yij = yi[j]; zij = zi[j]
        
        # all k != i,j OR I > J ???
        k = np.arange(len(ri[:])) > j  
        rik = ri[k] 
        xik = xi[k]; yik = yi[k]; zik = zi[k]

        # compute cos(theta_ijk) and rjk
        theta = np.arccos( (xij*xik + yij*yik + zij*zik) / (rij*rik) )
        theta *= 180/np.pi

        # add to this list of angles for this step
        angleStep.append(theta.tolist())

    # flatten list    
    angleStep = [item for sublist in angleStep for item in sublist]

    # add to total nested list
    angles.append(angleStep)


nBins = 90

bins = np.linspace(0, 180, nBins+1)
binCenters = (bins[1:] + bins[:-1]) / 2.0


##### histogram for time step of choice #####
#plt.hist(angles[500], nBins)



##### total time-averaged histogram #####
cumulativeAngles = np.zeros(nBins)
for i in xrange(nTimeSteps):
    for j in xrange(len(angles[i])):
        for k in xrange(nBins):
            if angles[i][j] < bins[k]:
                cumulativeAngles[k] += 1
                break

# normalize
cumulativeAngles /= nTimeSteps
plt.plot(binCenters, cumulativeAngles, 'g-')
plt.show()



##### customized average #####
"""Nevery = 10
Nrepeat = 10
Nfreq = 100
assert Nfreq % Nevery == 0
assert Nevery*Nrepeat <= Nfreq

# make indicies corresponding to the above values
# include the initial configuration also
nAverages = nTimeSteps / Nfreq
indicies = []; indicies.append([0])
for i in xrange(1,nAverages+1,1):
    end = i*Nfreq + Nevery
    start = end - Nevery*(Nrepeat)
    indicies.append(range(start, end, Nevery))

print indicies

cumulativeAngles = np.zeros((nAverages,nBins))
for i in xrange(nAverages):
    for timeStep in indicies[i]:
        for j in xrange(len(angles[timeStep])):
            for k in xrange(nBins):
                if angles[timeStep][j] < bins[k]:
                    cumulativeAngles[i][k] += 1
                    break
    cumulativeAngles[i] /= len(indicies[i])

plt.plot(binCenters, cumulativeAngles[1])
plt.show()"""
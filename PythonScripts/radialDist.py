import numpy as np
import matplotlib.pyplot as plt

with open('../Argon/Data/radialDist.dat', 'r') as inFile:
	
	# skip lines
    for _ in range(3):
		inFile.readline()

    timeStep, nBins = inFile.readline().split()
    timeStep = float(timeStep); nBins = float(nBins)
    
    bins = []; bins.append(0)
    distribution = []
    for line in inFile:
        words = line.split()
        bins.append(float(words[1]))
        distribution.append(float(words[2]))

bins = np.array(bins)
distribution = np.array(distribution)
binCenters = (bins[1:] + bins[:-1]) / 2.0

plt.plot(binCenters, distribution)
plt.show()




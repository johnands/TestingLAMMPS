import numpy as np
import matplotlib.pyplot as plt

with open('../Silicon/Data/radialDist.dat', 'r') as inFile:
	
	# skip lines
    for _ in range(3):
		inFile.readline()

    timeStep, nBins = inFile.readline().split()
    timeStep = float(timeStep); nBins = float(nBins)
    
    bins = []; bins.append(0.0)
    distribution = []
    for line in inFile:
        words = line.split()
        if len(words) != 4:
			break
        bins.append(float(words[1]))
        distribution.append(float(words[2]))

	i = 0
    for line in inFile:
        words = line.split()
        print words
        if len(words) != 4:
            i = 0
            continue
        distribution[i] += float(words[2])
        i += 1
		


bins = np.array(bins)
distribution = np.array(distribution) / 200
binCenters = (bins[1:] + bins[:-1]) / 2.0

plt.plot(binCenters, distribution)
plt.show()




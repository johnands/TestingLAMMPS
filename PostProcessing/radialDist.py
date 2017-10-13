import numpy as np
import matplotlib.pyplot as plt
import sys

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandParentDir = os.path.dirname(parentdir)
gGrandParentDir = os.path.dirname(grandParentDir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, grandParentDir)
sys.path.insert(2, gGrandParentDir)

import TensorFlow.Tools.matplotlibParameters as pltParams


def readRadialdist(filename):

    with open(filename, 'r') as inFile:
        """Read radial distribution dump from LAMMPS"""
    	
        	# skip lines
        for _ in range(3):
            inFile.readline()
    
        timeSteps = []
        timeStep0, nBins = inFile.readline().split()
        timeSteps.append(float(timeStep0)); 
        nBins = int(nBins)
        
        # read bins and first distribution
        bins = []; bins.append(0.0)
        distribution = []
        for line in inFile:
            words = line.split()
            if len(words) != 4:
                timeSteps.append(float(words[0]))
                break
            bins.append(float(words[1]))
            distribution.append(float(words[2]))
    
        	# read the other distributions
        for line in inFile:
            words = line.split()
            if len(words) != 4:
                timeSteps.append(float(words[0]))
                continue
            distribution.append(float(words[2]))
            
    return timeSteps, distribution, bins, nBins
		
  
  
def singleDistCompare(fileTarget, fileNN):
    
    timeSteps, distribution, bins, nBins = readRadialdist(fileTarget)
    timeStepsNN, distNN, binsNN, nBinsNN = readRadialdist(fileNN)
    
    assert(np.array_equal(timeSteps, timeStepsNN))
    assert(np.array_equal(bins, binsNN))
    
    bins = np.array(bins)
    distribution = np.array(distribution)
    distNN = np.array(distNN)
    binCenters = (bins[1:] + bins[:-1]) / 2.0
    
    # plot first distribution
    cut = 275
    plt.plot(binCenters[cut:], distribution[cut:], 'k--', binCenters[cut:], distNN[cut:], 'g-')
    plt.legend(['SW', 'NN'], prop={'size':18})
    plt.xlabel(r'$r \; [\mathrm{\AA{}}]$')
    plt.ylabel(r'$g(r)$')
    plt.tight_layout()
    #plt.savefig('../../Oppgaven/Figures/Results/radialDist.pdf')
    plt.show()
    
    
    
def timeDist(filename, write=False):
    
    timeSteps, distribution, bins, nBins = readRadialdist(filename)
    
    bins = np.array(bins)
    distribution = np.array(distribution)
    binCenters = (bins[1:] + bins[:-1]) / 2.0

    nTimeSteps = len(timeSteps)

    # plot total time-averaged distribution
    timeAveragedDist = np.zeros(nBins)
    for i in xrange(nTimeSteps):
        for j in xrange(nBins): 
            timeAveragedDist[j] += distribution[i*nBins + j]
     
    timeAveragedDist /= nTimeSteps

    if write: 
        with open('tmp/radialDist.txt', 'w') as outfile:
            for i in xrange(len(binCenters)):
                outfile.write('%g %g' % (binCenters[i], distribution[i]))
                outfile.write('\n')
        

    # plot time-averaged distribution
    plt.plot(binCenters, timeAveragedDist)
    plt.legend(['Radial distribution function'])
    plt.show()
    
    

if len(sys.argv) > 2:
    filenameTarget = sys.argv[1]
    filenameNN = sys.argv[2]
    singleDistCompare(filenameTarget, filenameNN)
    
else:
    filename = sys.argv[1]
    timeDist(filename)
    




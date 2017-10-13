import numpy as np


def pickSamples(filename, samples):
    
    neighbours = []
    i = 0
    j = 0
    nSamples = len(samples)
    with open(filename, 'r') as infile:
        for line in infile:
            if i == samples[j]:
                words = line.split()
                ineigh = []
                for word in words:
                    ineigh.append(float(word))
                neighbours.append(ineigh)
                j += 1
                if j == nSamples:
                    break
                print samples[j]
            i += 1
        
    return neighbours
    
def readSamples(filename):
    
    samples = []
    with open(filename, 'r') as infile:
        for line in infile:
            samples.append(int(line))
            
    return samples
        
    
atomID = 100
low = 10000
high = 13000
filename = '../TestNN/Data/Si/TrainingData/Bulk/L3T300N%d-%dA%d/neighbours.txt' % (low, high, atomID)
samplesFile  = 'Samples/samples%d-%dA%d.txt' % (low, high, atomID)


samples = readSamples(samplesFile)
print samples

neighbours = pickSamples(filename, samples)

write = True
if write:
    with open('Neighbours/neighbours%d-%dA%d.txt' % (low, high, atomID), 'w') as outfile:
        for i in neighbours:
            for j in i:
                outfile.write('%g ' % j)
            outfile.write('\n')


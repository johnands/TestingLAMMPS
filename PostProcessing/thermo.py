import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) > 1:
    plot = sys.argv[1]
else:
    plot = 'one'

def readFile(filename):
    
    with open(filename, 'r') as infile:
        
        # skip two comment lines
        infile.readline()
        infile.readline()
        
        temperature = []
        kineticEnergy = []
        potentialEnergy = []
        pressure = []
        for line in infile:
            words = line.split()
            temperature.append(float(words[1]))
            kineticEnergy.append(float(words[2]))
            potentialEnergy.append(float(words[3]))
            pressure.append(float(words[4]))
            
    temperature = np.array(temperature)
    kineticEnergy = np.array(kineticEnergy)
    potentialEnergy = np.array(potentialEnergy)
    pressure = np.array(pressure)
    
    return temperature, kineticEnergy, potentialEnergy, pressure
    
    
dirNameNN = '../TestNN/Data/SiO2/Thermo/Atoms2N1e4NoZeros/'
dirNameSW = '../Quartz/Data/Thermo/Atoms2N1e4/'

# write out README files
print "Content of SW folder:"
os.system('cat ' + dirNameSW + 'README.txt')
print "Content of NN folder:"
os.system('cat ' + dirNameNN + 'README.txt')

tempNN, kinNN, potNN, pressNN = readFile(dirNameNN + 'thermo.txt')
tempSW, kinSW, potSW, pressSW = readFile(dirNameSW + 'thermo.txt')

potFactor = np.average(potNN / potSW)
print "Potential energy factor: ", potFactor
#potNN /= potFactor

totalEnergyNN = kinNN + potNN
totalEnergySW = kinSW + potSW

numberOfSamples = len(tempNN)

print "Slope of energy drift: ", (totalEnergyNN[-1] - totalEnergyNN[0])/len(tempNN)

print 
print "Average temp NN: ", np.average(tempNN)
print "Average temp SW: ", np.average(tempSW)
print "Average kin NN: ", np.average(kinNN)
print "Average kin SW: ", np.average(kinSW)
print "Average pot NN: ", np.average(potNN)
print "Average pot SW: ", np.average(potSW)
print "Average tot NN: ", np.average(totalEnergyNN)
print "Average tot SW: ", np.average(totalEnergySW)

print
print "Std. dev. temp oscillations NN: ", np.std(tempNN)
print "Std. dev. temp oscillations SW: ", np.std(tempSW)


if plot == 'both':
    plt.subplot(2,2,1)
    plt.plot(tempNN, 'b-', tempSW, 'g-')
    plt.subplot(2,2,2)
    plt.plot(kinNN, 'b-', kinSW, 'g-')
    plt.subplot(2,2,3)
    plt.plot(potNN, 'b-', potSW, 'g-')
    plt.subplot(2,2,4)
    plt.plot(totalEnergyNN, 'b-', totalEnergySW, 'g-')
    plt.show()
    
elif plot == 'kin':
    plt.plot(kinNN - kinSW)
    plt.xlabel('Timestep')
    plt.ylabel('Kinetic energy')
    plt.show()
    
else: 
    plt.subplot(2,2,1)
    plt.plot(tempNN)
    plt.subplot(2,2,2)
    plt.plot(kinNN)
    plt.subplot(2,2,3)
    plt.plot(potNN)
    plt.subplot(2,2,4)
    plt.plot(totalEnergyNN)
    plt.show()
    



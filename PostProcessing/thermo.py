import numpy as np
import matplotlib.pyplot as plt
import sys


plotBoth = sys.argv[1]


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
    

tempNN, kinNN, potNN, pressNN = readFile('../TestNN/Data/Thermo/21.03-13.48.16/thermo.txt')
tempSW, kinSW, potSW, pressSW = readFile('../Silicon/Data/Thermo/21.03-14.05.53/thermo.txt')

totalEnergyNN = kinNN + potNN
totalEnergySW = kinSW + potSW

print potNN[0] / potSW[0]

numberOfSamples = len(tempNN)

print "Slope of energy drift: ", (totalEnergyNN[-1] - totalEnergyNN[0])/len(tempNN)

# compute averages
aveTempNN = sum(tempNN) / numberOfSamples
aveTempSW = sum(tempSW) / numberOfSamples
aveKinNN = sum(kinNN) / numberOfSamples
aveKinSW = sum(kinSW) / numberOfSamples
avePotNN = sum(potNN) / numberOfSamples
avePotSW = sum(potSW) / numberOfSamples
aveTotNN = sum(totalEnergyNN) / numberOfSamples
aveTotSW = sum(totalEnergySW) / numberOfSamples

print "Average temp NN: ", aveTempNN
print "Average temp SW: ", aveTempSW
print "Average kin NN: ", aveKinNN
print "Average kin SW: ", aveKinSW
print "Average pot NN: ", avePotNN
print "Average pot SW: ", avePotSW
print "Average tot NN: ", aveTotNN
print "Average tot SW: ", aveTotSW

if plotBoth == 'both':
    plt.subplot(2,2,1)
    plt.plot(tempNN, 'b-', tempSW, 'g-')
    plt.subplot(2,2,2)
    plt.plot(kinNN, 'b-', kinSW, 'g-')
    plt.subplot(2,2,3)
    plt.plot(potNN, 'b-', potSW, 'g-')
    plt.subplot(2,2,4)
    plt.plot(totalEnergyNN, 'b-', totalEnergySW, 'g-')
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
    



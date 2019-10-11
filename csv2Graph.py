import csv
import numpy
from matplotlib import pyplot
import requests

filePath = "geData.csv"
csvFile = open(filePath,"r")
reader = csv.reader(csvFile)
geData = []
for row in reader:
    if(row[4]!="Close"):
        geData.append(float(row[4]))

pyplot.figure()
pyplot.plot(geData)
pyplot.show()


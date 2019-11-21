import csv
import numpy
from matplotlib import pyplot
import requests

filePath = "data\\GE.csv"
isAscending = False

csvFile = open(filePath,"r")
reader = csv.reader(csvFile)
geData = []
for row in reader:
    if(row[4]!="close"):
        geData.append(float(row[4]))

if isAscending == False:
    geData.reverse()

pyplot.figure()
pyplot.plot(geData)
pyplot.show()


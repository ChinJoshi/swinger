import csv
import platform
import os
from glob import glob

INPUT_PATH = "../models/GEmodels"
isCSV = False
symbol = "GE"

subfolders = [f.path for f in os.scandir(INPUT_PATH) if f.is_dir() ]
lossDict = []
for folder in subfolders:
    logFP = None
    if platform.system()=="Windows":
        logFP = glob(folder+"\\*.log")
    else:
        logFP = glob(folder+"/*.log")
    with open(logFP[0]) as logFile:
        fileContents = csv.reader(logFile, delimiter=',')
        valLoss = []
        for index, row in enumerate(fileContents):
            if index%10==0 and index!=0:
                valLoss.append(float(row[2]))
        valLoss.sort()
        lossDict.append([valLoss[0],logFP[0]])

lossDict.sort()
csvfile=None
csvwriter=None
if isCSV:
    csvfile = open(symbol+"modelSort.csv","w",newline='')
    csvwriter = csv.writer(csvfile,delimiter=",")

for row in lossDict:
    print(row)
    if isCSV:
        csvwriter.writerow(row)

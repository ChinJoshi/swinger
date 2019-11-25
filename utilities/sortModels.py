import csv
import os
from glob import glob

INPUT_PATH = "E:\\GEModels"
isCSV = False
symbol = "GE"

subfolders = [f.path for f in os.scandir(INPUT_PATH) if f.is_dir() ]
lossDict = []
for folder in subfolders:
    logFP = glob(folder+"\\*.log")
    with open(logFP[0]) as logFile:
        fileContents = csv.reader(logFile, delimiter=',')
        valLoss = []
        for index, row in enumerate(fileContents):
            if index%10==0 and index!=0:
                valLoss.append(float(row[2]))
        valLoss.sort()
        lossDict.append([valLoss[0],logFP[0]])

lossDict.sort()

csvfile = open(symbol+"modelSort.csv","w",newline='')
csvwriter = csv.writer(csvfile,delimiter=",")

for row in lossDict:
    print(row)
    if isCSV:
        csvwriter.writerow(row)
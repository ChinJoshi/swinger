import os
import csv
from matplotlib import pyplot as plt

INPUT_LOG_PATH = "C:\\Users\\Chinmaya Joshi\\Downloads\\ml Project\\GEmodels\\GE_Model_B256_T60_L1N500_L2D0.3_L3N500_L4D0.4\\GE_Model_B256_T60_L1N500_L2D0.3_L3N500_L4D0.4.log"
loss = []
val_loss =[]
with open(INPUT_LOG_PATH) as logFile:
    csv_reader = csv.reader(logFile, delimiter=",")
    for i,row in enumerate(csv_reader):
        if(i==0):
            continue
        loss.append(float(row[1]))
        val_loss.append(float(row[2]))

plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title('Losses')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

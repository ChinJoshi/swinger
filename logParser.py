import os
import csv
from matplotlib import pyplot as plt

INPUT_LOG_PATH = "D:\\GEModels\\GE_Model_B256_T60_CUDA\\GE_Model_B256_T60_CUDA.log"
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

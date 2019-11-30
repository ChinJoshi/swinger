import os
import csv
from matplotlib import pyplot as plt

INPUT_LOG_PATH = "../models/GEmodels/GE_lstm_500_dropout_0.3_lstm_400_dropout_0.3_lstm_300_dropout_0.3_lstm_200_dropout_0.3_lstm_100_dropout_0.3_lstm_100_dropout_0.3_dense_100_dense_1/GE_lstm_500_dropout_0.3_lstm_400_dropout_0.3_lstm_300_dropout_0.3_lstm_200_dropout_0.3_lstm_100_dropout_0.3_lstm_100_dropout_0.3_dense_100_dense_1.log"
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

import numpy as np
import platform
import os
import sys
import time
import pandas as pd 
import glob
import tensorflow
from tensorflow import keras
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


params = {
    "batch_size": 256,
    "epochs": 250,
    "lr": 0.00010000,
    "time_steps": 60
}

train_cols = ['open','high','low','close','adjusted_close','volume','dividend_amount','split_coefficient','RSI','SMA','EMA','Real Lower Band','Real Middle Band','Real Upper Band']
TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
DATA_PATH = "data\\GE.csv"
symbol = "GE"
isAscending = False

layer_1_neurons = 100
layer_2_dropout = 0.3
layer_3_neurons = 100
layer_4_dropout = 0.4
layer_5_denseNeurons = 100

def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")

#Make dataset fit into the timestep
def trim_dataset(mat,batch_size):
    no_of_rows_drop = mat.shape[0]%batch_size   #determine how many extra rows you have that don't fit into a multiple of batch
    if no_of_rows_drop > 0:
        print("Rows to drop: "+ str(no_of_rows_drop))
        tempMat = mat[:-no_of_rows_drop]        #cuts off last rows, so it should work (the most recent data is cut off so this function needs to be fixed)
        print("Mat shape after cut: " + str(tempMat.shape[0]))
        return tempMat
        #return mat[:-no_of_rows_drop]   #cut that bitch off
    else:
        return mat


def build_timeseries(mat, y_col_index):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output 
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS   #determine number of inputs that have an output because you'll not have data timesteps forward once your index goes past len(mat.shape[0]) - TIME_STEPS
    dim_1 = mat.shape[1]    #number of columns in dataset
    x = np.zeros((dim_0, TIME_STEPS, dim_1))    #create an array with dimensions of viable inputs, the time step, the cols in the dataset
    y = np.zeros((dim_0,))  #create an output array to correspond to input array

    for i in range(dim_0):  #for each viable input
        x[i] = mat[i:TIME_STEPS+i]  #set the input array to a set of data the length of a timestep
        y[i] = mat[TIME_STEPS+i, y_col_index]   #set output array that correalates with inout array to that data located in the y_col_index of the data
    #print("length of time-series i/o",x.shape,y.shape)
    return x, y

stime = time.time()
df_ge = pd.read_csv(DATA_PATH, engine='python')    #read the csv file into a pandas dataset

print(df_ge.head(5))

if (isAscending==False):
    print("\n########After Reversal########\n")
    df_ge = df_ge.iloc[::-1]
    print(df_ge.head(5))


df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)   #split the data into a traing set and validation set (kind of retarded because the most recent data becomes test set aka braindead)
print("Train--Test size", len(df_train), len(df_test))

# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values   #grabs the data only(no headers) specified in train_cols
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)   #Scale the training set and determine the scaling factors
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])    #Scale the the validation set using the precalculated scale factor
print(x_train[0])
print("Deleting unused dataframes of total size(KB)",(sys.getsizeof(df_ge)+sys.getsizeof(df_train)+sys.getsizeof(df_test))//1024)

del df_ge
del df_test
del df_train
del x

print("Are any NaNs present in train/test matrices?",np.isnan(x_train).any(), np.isnan(x_train).any())


x_t, y_t = build_timeseries(x_train, 3) #Build training dataset with valid points(those who have 60 days preceeding them)
x_t = trim_dataset(x_t, BATCH_SIZE) #Make input data fit in the batch size
y_t = trim_dataset(y_t, BATCH_SIZE) #Make output data fit in the batch size
print("Batch trimmed size",x_t.shape, y_t.shape)

x_temp, y_temp = build_timeseries(x_test, 3)    #Build validation dataset
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)      #split validation set further into 2 pieces,  evaluation set and test set (makes no sense)   
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)      #split validation set further into 2 pieces,  evaluation set and test set (makes no sense)

print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)

MODEL_NAME = symbol+"_Model_B" + str(params["batch_size"]) + "_T" + str(params["time_steps"])+ "_L1N" + str(layer_1_neurons) + "_L2D" + str(layer_2_dropout) + "_L3N" + str(layer_3_neurons) + "_L4D" + str(layer_4_dropout)

OUTPUT_PATH=None
if platform.system()=="Windows":
    OUTPUT_PATH = symbol+"models\\" + MODEL_NAME
else:
    OUTPUT_PATH = symbol+"models/" + MODEL_NAME


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
else:
    print("Output directory exists,exiting to prevent overwriting")
    quit()

lstm_model = keras.Sequential()
lstm_model.add(keras.layers.CuDNNLSTM(layer_1_neurons, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), stateful=True, return_sequences=True, kernel_initializer='random_uniform'))
lstm_model.add(keras.layers.Dropout(layer_2_dropout))
lstm_model.add(keras.layers.CuDNNLSTM(layer_3_neurons))
lstm_model.add(keras.layers.Dropout(layer_4_dropout))
lstm_model.add(keras.layers.Dense(layer_5_denseNeurons,activation='relu'))
lstm_model.add(keras.layers.Dense(1,activation='sigmoid'))

optimizer = keras.optimizers.RMSprop(lr=params["lr"])
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

mcp = keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH, MODEL_NAME+".{epoch:02d}-{val_loss:.5f}.hdf5"), monitor='val_loss', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=10)
csv_logger=None
if platform.system()=="Windows":
    csv_logger = keras.callbacks.CSVLogger(OUTPUT_PATH+"\\"+MODEL_NAME+".log", append=True)
else:
    csv_logger = keras.callbacks.CSVLogger(OUTPUT_PATH+"/"+MODEL_NAME+".log", append=True)

lstm_model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE)), callbacks=[mcp,csv_logger])




y_pred = lstm_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])
y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_test_t)
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

# Visualize the prediction
plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()
#plt.savefig(os.path.join(OUTPUT_PATH, 'pred_vs_real_BS'+str(BATCH_SIZE)+"_"+time.ctime()+'.png'))
print_time("program completed ", stime)
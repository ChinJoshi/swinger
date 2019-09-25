import numpy as np
import os
import sys
import time
import pandas as pd 
import glob
import tensorflow
from tensorflow import keras

# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# import talos as ta

params = {
    "batch_size": 258,  # 20<16<10, 25 was a bust
    "epochs": 300,
    "lr": 0.00010000,
    "time_steps": 60
}

continuing = False
INPUT_PATH = "D:\\GEModels"
DATA_FILE = "geData.csv"
MODEL_NAME = "GE_Model_B" + str(params["batch_size"]) + "_T" + str(params["time_steps"])+ "_CUDA"
OUTPUT_PATH = "D:\\GEModels\\" + MODEL_NAME
TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
stime = time.time()

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
elif continuing:
    print("Resuming Training From Last Checkpoint")
else:
    print("Output directory exists,exiting to prevent overwriting")
    quit()

def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")

#Make dataset fit into the timestep
def trim_dataset(mat,batch_size):
    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
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
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    print("dim_0",dim_0)
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y


stime = time.time()
print(os.listdir(INPUT_PATH))
df_ge = pd.read_csv(INPUT_PATH+"\\"+DATA_FILE, engine='python')
print(df_ge.shape)
print(df_ge.columns)
print(df_ge.head(5))

#df_ge = process_dataframe(df_ge)
print(df_ge.dtypes)
train_cols = ["Open","High","Low","Close","Volume"]
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
print("Train--Test size", len(df_train), len(df_test))

# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

print("Deleting unused dataframes of total size(KB)",(sys.getsizeof(df_ge)+sys.getsizeof(df_train)+sys.getsizeof(df_test))//1024)

del df_ge
del df_test
del df_train
del x

print("Are any NaNs present in train/test matrices?",np.isnan(x_train).any(), np.isnan(x_train).any())
x_t, y_t = build_timeseries(x_train, 3)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
print("Batch trimmed size",x_t.shape, y_t.shape)


def create_model():
    lstm_model = keras.Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(keras.layers.CuDNNLSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                        stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(keras.layers.Dropout(0.4))
    lstm_model.add(keras.layers.CuDNNLSTM(60))
    lstm_model.add(keras.layers.Dropout(0.4))
    lstm_model.add(keras.layers.Dense(20,activation='relu'))
    lstm_model.add(keras.layers.Dense(1,activation='sigmoid'))
    optimizer = keras.optimizers.RMSprop(lr=params["lr"])
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model

x_temp, y_temp = build_timeseries(x_test, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)

model=None

def sortSecond(val):
    return val[1]

if continuing:
    model_files = glob.glob(OUTPUT_PATH+"\\*.hdf5")
    creationTimes = []
    for model_file in model_files:
        file_time = []
        file_time.append(model_file)
        file_time.append(os.path.getctime(model_file))
        creationTimes.append(file_time)
    creationTimes.sort(key=sortSecond)
    print(creationTimes)
    model = keras.models.load_model(creationTimes[-1][0])
    trueEpoch = int(creationTimes[-1][0].split("-")[0].split(".")[1])

    csv_logger = keras.callbacks.CSVLogger(OUTPUT_PATH+"\\"+MODEL_NAME+".log", append=True)
    mcp = keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH,
                          MODEL_NAME +"."+ str(trueEpoch) + "+" + "{epoch:02d}-{val_loss:.5f}.hdf5"), monitor='val_loss', verbose=1,
                          save_best_only=False, save_weights_only=False, mode='auto', period=10)
    history = model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                        trim_dataset(y_val, BATCH_SIZE)), callbacks=[mcp,csv_logger])    
    
else:
    print("Building model...")
    #print("checking if GPU available", keras.backend.tensorflow_backend._get_available_gpus())
    model = create_model()
    
    #Stop training model when quality monitored hasn't importved for patience number of epochs
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=40, min_delta=0.0001)
    
    mcp = keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH,
                          MODEL_NAME+".{epoch:02d}-{val_loss:.5f}.hdf5"), monitor='val_loss', verbose=1,
                          save_best_only=False, save_weights_only=False, mode='auto', period=10)

    # Not used here. But leaving it here as a reminder for future
    r_lr_plat = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, 
                                  verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    
    csv_logger = keras.callbacks.CSVLogger(OUTPUT_PATH+"\\"+MODEL_NAME+".log", append=True)
    
    """ history = model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                        trim_dataset(y_val, BATCH_SIZE)), callbacks=[es, mcp, csv_logger]) """
    history = model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                        trim_dataset(y_val, BATCH_SIZE)), callbacks=[mcp,csv_logger])
    
    print("saving model...")




# model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE
y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])

# convert the predicted value to range of real data
y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
# min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
# min_max_scaler.inverse_transform(y_test_t)
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

# Visualize the training data
from matplotlib import pyplot as plt
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#plt.savefig(os.path.join(OUTPUT_PATH, 'train_vis_BS_'+str(BATCH_SIZE)+"_"+time.ctime()+'.png'))

y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
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
from matplotlib import pyplot as plt
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
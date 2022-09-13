

#%%
import os
import pandas as pd
import numpy as np
import tensorflow as ts
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error , mean_squared_error
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("C:/Users/utilisateur/OneDrive - Simplonformations.co/Bureau/Programmations/LSTM_polution/lstm_polution/LSTM-Multivariate_pollution.csv")
# %%
# Encode Wind direction
df.wnd_dir.unique()
# %%
def wind_encode(s):
    if s == "SE":
        return 1
    elif s == "NE":
        return 2
    elif s == "NW":
        return 3
    else:
        return 4

df["wind_dir"] = df["wnd_dir"].apply(wind_encode)
del df["wnd_dir"]

df.head()
# %%
# Can't scale Date, i drop it
df.drop(["date"], inplace = True, axis = 1)
df.head()
# %%
#Scaled data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
scaled = pd.DataFrame(scaled)
scaled.head()

# %%
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)
    cols, names = list(), list()

    #input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j+1, i)) for j in range(n_vars)]

    # Forecast sequence (t, t+1,...,t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j+1, i)) for j in range(n_vars)]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names 

    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg
# %%
reframed = series_to_supervised(scaled, 1, 1)
print(reframed.shape)
# %%
# droping columns we don't want to predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis = 1, inplace = True)
print(reframed.head())
# %%
# split train and validation data
#Eache rows is 1 hour, each year = 24 hours * 365 days
year = 24 * 365
values = reframed.values

#train on forst 3 years and validate w/ rst
train = values[:(3 * year), :]
val = values[(3 * year):, :]

#split training and validation inte feature and target
X_train, y_train = train[:, :-1], train[:, -1]
X_val, y_val = val[:, :-1], val[:, -1]

#reshape input 3 dim
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# %%
def create_model():
    opti = Adam(learning_rate=0.01)

    model = Sequential()
    model.add(LSTM(256, input_shape = (X_train.shape[1], X_train.shape[2]), activation = "relu", return_sequences=True))
    model.add(LSTM(150, activation = "relu", return_sequences=True))
    model.add(LSTM(75, activation = "relu", return_sequences=True))
    model.add(Dense(200))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(75))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(1))


    model.compile(loss = "mse",
                        optimizer = opti,
                        metrics = ["accuracy"])

    return model
 #%%
#Create ModelChekPoint
checkpoint_path = "training_1/cp-{epoch:04d}.cpkt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 170

cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                save_weights_only=True,
                verbose = 0,
                save_freq= 5 * batch_size)

# %%

model = create_model()
history = model.fit(X_train, y_train, epochs = 100,
            batch_size = 170, validation_split = 0.2, 
            verbose = 2, shuffle = False, validation_data=(X_val, y_val),
            )

# %%
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
poll = np.array(df["pollution"])

poll_mean = poll.mean()
poll_std = poll.std()

y_val_true = val[:,8]
val_pred = model.predict(X_val).ravel()

y_val_true = y_val_true*poll_std + poll_mean
val_pred = val_pred*poll_std + poll_mean
from matplotlib import pyplot as plt

plt.figure(figsize=(15,6))
plt.xlim([1000,1250])
plt.ylabel("ppm")
plt.xlabel("hrs")
plt.plot(val_pred, c = "r", alpha = 0.90, linewidth = 2.5)
plt.plot(y_val_true, c = "b", alpha = 0.75)
plt.title("Validation Data Prediction", fontsize=20)
plt.legend(['Predicted Values', 'True Values'], fontsize='xx-large')
plt.show()

#%%
#Save model
model.save("LSTM.h5")
# %%
# Import Test data
test_df = df.copy()
#test_df["wind_dir"] = test_df["wind_dir"].apply(wind_encode)
#del test_df["win_dir"]

values_test = test_df.values
values_test = values_test.astype("float32")

scaler1 = MinMaxScaler()
scaled_test = scaler1.fit_transform(values_test)

reframed_test = series_to_supervised(scaled_test, 1, 1)

reframed_test.drop(reframed_test.columns[[9,10,11,12,13,14,15]], axis = 1, inplace = True)

values_test1 = reframed_test.values
test_x, test_y = values_test1[:, :-1], values_test1[:, -1]
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
val_pred4 = model.predict(test_x).ravel()

#inverse scaling the ouptu, for better visuel interpretation
test_y = test_y * poll_std + poll_mean
val_pred4 = val_pred4 * poll_std + poll_mean

# %%
plt.figure(figsize=(18,5.5))
plt.ylabel("ppm")
plt.xlabel("hrs")
plt.plot(test_y, c = "r", alpha = 0.90)
plt.plot(val_pred4, c = "darkblue", alpha = 0.75)
plt.title("Testing Data Prediction")
plt.legend(['Predicted Values', 'True Values'], fontsize='x-large')
plt.show()
# %%

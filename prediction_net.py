# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:44:20 2019

@author: tanma
"""

import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, SpatialDropout1D, GRU, LSTM,Conv1D, concatenate, Dense
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Bidirectional 
from bearing_cal import calculate_initial_compass_bearing as cal

copy = pd.read_csv("new_dat.csv")
id_subset = [30,31,41,37962,27]
speed = []
latitude = []
longitude = []
time = []

for i in id_subset:
    speed.append(list(map(float,list(copy['speed'][copy['id_x'] == i].values))))
    latitude.append(list(map(float,list(copy['latitude'][copy['id_x'] == i].values))))
    longitude.append(list(map(float,list(copy['longitude'][copy['id_x'] == i].values))))
    time.append(list(map(str,list(copy['time_y'][copy['id_x'] == i].values))))

pnew = pd.DataFrame(columns = ['speed','longitude','latitude','direction','time'])
pnew.speed = [item for sublist in speed for item in sublist]
pnew.latitude = [item for sublist in latitude for item in sublist]
pnew.longitude = [item for sublist in longitude for item in sublist]
pnew.time = [item for sublist in time for item in sublist]

direction = [0]
for i in range(1,len(pnew)):
    direction.append(cal(pnew.iloc[i-1,2],pnew.iloc[i,2],pnew.iloc[i-1,1],pnew.iloc[i,1]))

pnew.direction = direction
pnew.to_csv('wassup.csv')
pnew = pd.read_csv('upscaled_new_dat1.csv')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()

	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	agg = pd.concat(cols, axis=1)
	agg.columns = names

	if dropnan:
		agg.dropna(inplace=True)
	return agg

agg = series_to_supervised(pnew)
values = agg.values

timestep = 1
train = values[:2000000,:]

scaler = StandardScaler()
train = scaler.fit_transform(train)

X_train = []
y_train = []
for i in range(timestep, len(pnew)-1):
    X_train.append(train[i-timestep:i, :len(pnew.columns)])
    y_train.append(train[i-timestep, [len(pnew.columns),-1]])
X_train, y_train = np.array(X_train), np.array(y_train)

inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
x = SpatialDropout1D(0.2)(inp)
x = LSTM(256,dropout=0.1,recurrent_dropout=0.1, return_sequences = True)(x)
x = Conv1D(64, kernel_size = 3, padding = "same", kernel_initializer = "glorot_uniform")(x)
x = LSTM(128,dropout=0.1,recurrent_dropout=0.1, return_sequences = True)(x)
x = Conv1D(32, kernel_size = 3, padding = "same", kernel_initializer = "glorot_uniform")(x)
max_pool = GlobalMaxPooling1D()(x)
x = Dense(1024)(max_pool)
preds = Dense(len(pnew.columns)-2)(x)

model = Model(inp,preds)

model.compile(loss = 'mse', optimizer = 'nadam')
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks=[checkpoint]
model.fit(X_train,y_train,epochs = 10,batch_size = 32, callbacks = callbacks)

y_pred = model.predict(X_train)
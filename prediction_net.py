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
from keras.layers import CuDNNLSTM, CuDNNGRU
from bearing_cal import calculate_initial_compass_bearing as cal
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt

copy = pd.read_csv("new_dat.csv")
id_subset = [30,31,41,37962,27]
id_ = 30
speed = []
latitude = []
longitude = []
time = []
track_id = []

for i in id_subset:
    speed.append(list(map(float,list(copy['speed'][copy['id_x'] == i].values))))
    latitude.append(list(map(float,list(copy['latitude'][copy['id_x'] == i].values))))
    longitude.append(list(map(float,list(copy['longitude'][copy['id_x'] == i].values))))
    time.append(list(map(str,list(copy['time_y'][copy['id_x'] == i].values))))
    track_id.append(list(map(str,list(copy['id_x'][copy['id_x'] == i].values))))

pnew = pd.DataFrame(columns = ['speed','longitude','latitude','direction','time','track_id'])
pnew.speed = [item for sublist in speed for item in sublist]
pnew.latitude = [item for sublist in latitude for item in sublist]
pnew.longitude = [item for sublist in longitude for item in sublist]
pnew.time = [item for sublist in time for item in sublist]
pnew.track_id = [item for sublist in track_id for item in sublist]

def get_bearing(lat1,lat2,long1,long2):
    brng = Geodesic.WGS84.Inverse(lat1,long1,lat2,long2)['azi1']
    return brng

direction = [0]
for i in range(1,len(pnew)):
    direction.append(get_bearing(pnew.iloc[i-1,2],pnew.iloc[i,2],pnew.iloc[i-1,1],pnew.iloc[i,1]))

pnew.direction = direction
pnew.to_csv('wassup.csv')
pnew = pd.read_csv('no.csv')
pnew = pnew[pnew.track_id == id_]
copy = pnew.drop(['time','track_id','Unnamed: 0'],axis = 1)
pnew = pnew.drop(['time','track_id','Unnamed: 0','longitude','latitude'],axis = 1)

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

whatever = pnew.iloc[:,0].values.reshape((-1,3581)).flatten() + np.random.randint(low = -5,high = 5,size = (pnew.shape[0]))
pnew['speed'] = whatever
scaler = StandardScaler()
what = scaler.fit_transform(pnew)
agg = series_to_supervised(what)
values = agg.values

timestep = 1
train = values[:2000000,:]

X_train = []
y_train = []
for i in range(timestep, len(pnew)-1):
    X_train.append(train[i-timestep:i, :len(pnew.columns)])
    y_train.append(train[i-timestep, len(pnew.columns):])
X_train, y_train = np.array(X_train), np.array(y_train)

inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
x = CuDNNLSTM(256, return_sequences = True)(inp)
y = CuDNNGRU(128, return_sequences = True)(inp)
x = concatenate([x,y])
x = SpatialDropout1D(0.2)(x)
x = Conv1D(64, kernel_size = 3, padding = "same")(x)
max_pool = GlobalMaxPooling1D()(x)
avg_pool = GlobalAveragePooling1D()(x)
x = concatenate([max_pool,avg_pool])
x = Dense(1024)(x)
preds = Dense(len(pnew.columns))(x)

model = Model(inp,preds)

model.compile(loss = 'mse', optimizer = 'nadam')
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks=[checkpoint]
model.fit(X_train,y_train,epochs = 20,batch_size = 1, callbacks = callbacks, validation_split = 0.25)

y_pred = model.predict(X_train)
model.save('model.h5')

plt.plot(y_pred[5:,0],y_pred[5:,1], color = 'red')
plt.plot(y_train[5:,0],y_train[5:,1], color = 'blue')
plt.show()

vector = np.array(copy.iloc[5,:].values)
x = model.predict(np.reshape(vector[[0,3]],(1,1,2)))
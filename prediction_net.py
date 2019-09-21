# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:44:20 2019

@author: tanma
"""

import pandas as pd
from keras.models import Model
from keras.layers import Input, SpatialDropout1D, GRU, Conv1D, concatenate, Dense
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Bidirectional 

copy = pd.read_csv("new_dat.csv")
data = copy[['speed','longitude','latitude']]

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

agg = series_to_supervised(data)
values = agg.values

timestep = 1
track_ids = len(copy['id_x'].unique())
train = values[:(track_ids-2)*90, :]
test = values[(track_ids-2)*90:track_ids*90, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0]//timestep, 90, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0]//timestep, 90, test_X.shape[1]))

inp = Input(shape=(train_X.shape[1],train_X.shape(2)))
x = SpatialDropout1D(0.2)(inp)
x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
preds = Dense(1)(x)

model = Model(inp,preds)
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:08:42 2019

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
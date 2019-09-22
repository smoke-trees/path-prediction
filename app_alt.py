# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:02:15 2019

@author: tanma
"""

from dynamic_predictor import DynamicPredictor

d = DynamicPredictor()


import rl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

data = pd.read_csv('no.csv')
data = data.drop(['Unnamed: 0'],axis = 1)
subset = data.loc[data["track_id"] == 41]
vector = subset.iloc[-10,:-1]
speed = [vector[1]]
direction = [vector[4]]
newcoords = [[vector[3],vector[2]]]

for i in range(10):
    x = rl.speed(vector)
    y = rl.direction(vector)
    z = d.feed(x,y,subset.iloc[-10+i:,-1])
    print(z)
    speed.append(x)
    direction.append(y)
    newcoords.append(z)
    vector = [x,z[1],z[0],y]
    
comp_df = pd.DataFrame()

comp_df['lat_pred'] = np.array(newcoords)[1:,0]
comp_df['long_pred'] = np.array(newcoords)[1:,1]
comp_df['lat_actual'] = np.array(subset.iloc[-10:,2])
comp_df['long_actual'] = np.array(subset.iloc[-10:,1])

comp_df['var_lat'] = comp_df['lat_pred'] - comp_df['lat_actual']
comp_df['var_long'] = comp_df['long_pred'] - comp_df['long_actual']

print(comp_df.head())

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:07:45 2019

@author: tanma
"""

import rl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

data = pd.read_csv('no.csv')
data = data.drop(['time','Unnamed: 0'],axis = 1)
subset = data.loc[data["track_id"] == 30]
speed = [subset.iloc[0,:].values[0]]
direction = [subset.iloc[0,:].values[3]]
newcoords = [[subset.iloc[0,:].values[2],subset.iloc[0,:].values[1]]]
vector = []
for i in subset.values:
    x = rl.speed(i).tolist()[0][0]
    y = rl.direction(i).tolist()[0][0]
    z = list(rl.NewCoords(i))[:2]
    speed.append(x)
    direction.append(y)
    newcoords.append(z)
    vector.append([x,z[1],z[0],y])
    
comp_df = pd.DataFrame(vector, columns = ['speed','longitude','latitude','bearing'])
new_angle = [i%360 for i in comp_df['bearing'].values]
comp_df['bearing'] = new_angle
subset = subset.drop('track_id',axis = 1)
comp_df.to_csv('comp_df.csv')
subset.to_csv('subset.csv')
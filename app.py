# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 05:21:14 2019

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
vector = subset.iloc[-10,:-1]
speed = [vector[0]]
direction = [vector[3]]
newcoords = [[vector[2],vector[1]]]


for i in range(10):
    x = rl.speed(vector)
    y = rl.direction(vector)
    z = list(rl.NewCoords(vector))[:2]
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
    

ax = plt.plot(subset.iloc[3000:-10,1],subset.iloc[3000:-10,2], color='red')
plt.plot(np.array(newcoords)[:,1],np.array(newcoords)[:,0], color='blue')
plt.plot(comp_df['long_actual'],comp_df['lat_actual'],color='black')
plt.show()  

def plot_on_map(latlon):
    import folium 
    mapit = folium.Map( location=[subset.iloc[-10,2],subset.iloc[-10,1]], zoom_start=6 )
    for coord in range(len(latlon)):
        if coord >4:
            folium.Marker( location=[ latlon[coord][0], latlon[coord][1]], fill_color='#43d9de', radius=8 ).add_to( mapit)
        else:
            folium.Marker( location=[ latlon[coord][0], latlon[coord][1]], fill_color='#3186cc', radius=8 ).add_to( mapit)
    mapit.save('map.html')
    
cords = list(zip(list(comp_df['lat_pred']),list(comp_df['long_pred'])))


plot_on_map(cords + list(zip(list(comp_df['lat_actual']),list(comp_df['long_actual']))))

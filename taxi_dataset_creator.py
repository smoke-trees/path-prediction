#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:57:48 2019

@author: entropy
"""

import glob
import pandas as pd 

data_list_taxi = []
for i in glob.glob('taxi/A0000'):
    data = []
    for j in open(i).readlines():
        data.append(j[:-1].split(';'))
    data_list_taxi.append(data)

print(len(data_list_taxi))
taxi = pd.DataFrame(data_list_taxi[0],columns = ['id','timestamp','latitude','longitude','speed','direction'])

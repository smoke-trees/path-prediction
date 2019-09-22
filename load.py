# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 02:32:19 2019

@author: tanma
"""

import tensorflow as tf
from keras.models import load_model   

def init(model_name):
    model = load_model(model_name)
    graph = tf.get_default_graph()
    
    return model,graph
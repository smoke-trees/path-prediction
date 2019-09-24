import numpy as np
import random
import math

from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dense
import keras.backend as K
import load as l
from geopy import Point
from geopy.distance import VincentyDistance
import time

SPEED_MODEL = 'speed.h5'
DISTANCE_MODEL = 'direction.h5'

speed_model = load_model(SPEED_MODEL)
distance_model = load_model(DISTANCE_MODEL)

def speed(vector):
    x = speed_model.predict(np.reshape(vector[1],(1,1,1)))
#    reward = K.tanh(1/math.sqrt((newCoords[0]-x[0])**0.5 - (newCoords[1]-x[1])**0.5))*10

    return x

def direction(vector):

    x = distance_model.predict(np.reshape(vector[3],(1,1,1)))
#    reward = K.tanh(1/math.sqrt((newCoords[0]-x[0])**0.5 - (newCoords[1]-x[1])**0.5))*10
    return x

def NewCoords(vector,timeDelay = 1/3600):
    speed = vector[0]
    direction = vector[3]
    distance_km = speed * timeDelay
    distance_miles = distance_km * 0.621371
    return VincentyDistance(miles=distance_miles).destination(Point(vector[2],vector[1]),direction)

model=Sequential()
model.add(Dense(24,input_dim=2,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(2))
model.compile(optimizer='nadam',loss='mse')

gamma = 1.0
epsilon = 1.0
m = []
x_actual, y_actual = 30, 50


for i in range(50):
    state = [x_actual,y_actual]
    state = np.array([state])
    for t in range(50):
        if np.random.rand() <= epsilon:
            action = [random.randrange(0,55),random.randrange(0,360)]
        else:
            action = np.argmax(model.predict(state))
        
        
        #next_state,reward,done,observation = env.step(action)
        next_state, reward = NewCoords(state,action[0],action[1],1)
        
        next_state = np.array([next_state])

        tot = reward + gamma * np.max(model.predict(next_state))
        p = model.predict(state)[0]
        p[action] = tot
        model.fit(state, p.reshape(-1, 2), epochs=1, verbose=0)
        m.append((state,action,reward,next_state,done))
        state = next_state
        if done:
            print("Episode : {}, x_pred: {}, y_pred: {}".format(i,state[0],state[1]))
            break
        if len(m)==50000:
            del m[:5000]
    if epsilon > 0.01:
        epsilon *= 0.999
    if len(m) > 64:
        for state, action, reward, next_state, done in random.sample(m,64):
            tot=reward
            if not done:
              tot=reward + gamma * np.max(model.predict(next_state))

            p = model.predict(state)[0]
            p[action] = tot
            model.fit(state,p.reshape(-1,2), epochs=1, verbose=0)

for i in range(20):
    state = env.reset()
    i = 0
    while i != 10:
        env.render()
        action = np.argmax(model.predict(np.array([state])))
        #next_state, reward, done, observation = env.step(action)
        next_state, reward = NewCoords(state,action[0],action[1],1)
        state = next_state


import math


def calculateNewCoords(coords, speed, angle, timeDelay):
    newCoords = coords

    #convert new coords to mod 180, 360
    newCoords[0] += 90
    newCoords[1] += 360

    newCoords[0] += (speed * math.cos(math.pi * angle / 180) * timeDelay) % 360
    newCoords[1] = (newCoords[1] + speed * math.sin(math.pi * angle / 180) * timeDelay) % 360

    if newCoords[0] > 180:
        newCoords[0] = 360 - newCoords[0]
    
    return [newCoords[0] - 90, newCoords[1] - 180]



print(calculateNewCoords([89, 179], 400, 45, 1))


    
    

    
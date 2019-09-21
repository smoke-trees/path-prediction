
def calculateNewCoords(coords, speed, direction, timeDelay):
    newCoords = coords

    #convert new coords to mod 180, 360
    newCoords[0] += 90
    newCoords[1] += 360

    newCoords[0] += (speed * direction[0] * timeDelay) % 360
    newCoords[1] = (newCoords[1] + speed * direction[1] * timeDelay) % 360

    if newCoords[0] > 180:
        newCoords[0] = 360 - newCoords[0]
    
    return [newCoords[0] - 90, newCoords[1] - 180]



print(calculateNewCoords([89, 179], 400, [1, 1], 1))


    
    

    
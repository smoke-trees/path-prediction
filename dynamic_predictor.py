import math
from utils import calculateNewCoords


class DynamicPredictor:
    def __init__(self):
        self.speed_buffer = []
        self.direction_buffer = []
        self.time_stamps = []
        self.speed_grads = []
        self.direction_grads = []
        self.avg_speed_grad = 0
        self.avg_direction_grad = 0
        self.mem_size = 10
        self.max_speed = 0
        self.max_direction = 0
        # intention vector calculated by dynamic regression
        self.intention_angle = 0
        self.alpha = 0.1
        self.beta = 0.05
        self.p_lr = 0.1
        
    def sigmoid(self, val):
        return (1 / (1 + math.exp(-val)))

    def learn(self, pred_angle):
        if len(self.direction_buffer) < 2:
            return
        if math.fabs(self.direction_buffer[-1] - self.intention_angle) < math.fabs(self.direction_buffer[-1] - pred_angle):
            self.beta += self.p_lr
        else:
            self.beta -= self.p_lr
        if self.beta > 1:
            self.beta = 1
        if self.beta < 0:
            self.beta = 0


    def eval_grads(self):
        if len(self.time_stamps) < 2:
            return

        for i in range(len(self.time_stamps) - 1):
            self.avg_speed_grad += (self.speed_buffer[i+1] - self.speed_buffer[i])
            self.avg_direction_grad = (self.direction_buffer[i+1] - self.direction_buffer[i])
        
        self.avg_speed_grad /= (self.time_stamps[-1] - self.time_stamps[0])
        self.avg_direction_grad /= (self.time_stamps[-1] - self.time_stamps[0])

    def update_intention(self):
        if len(self.time_stamps) == 0:
            return
        self.intention_angle = self.alpha * self.direction_buffer[-1] + (1 - self.alpha) * self.intention_angle

    def feed(self, speed, direction, time_stamp):
        if speed > self.max_speed:
            self.max_speed = speed
        if direction > self.max_direction:
            self.max_direction = direction

        self.time_stamps.append(time_stamp)
        self.speed_buffer.append(speed)
        self.direction_buffer.append(direction)
        if len(self.time_stamps) > self.mem_size:
            del self.time_stamps[0]
            del self.speed_buffer[0]
            del self.direction_buffer[0]
        
        self.update_intention()



    def predict(self, time_elapsed):
        if len(self.time_stamps) < 2:
            return

        self.eval_grads()
        # returns new speed and direction
        predicted_direction = self.direction_buffer[-1] + (self.avg_direction_grad * time_elapsed)
        predicted_speed = self.speed_buffer[-1] + (self.avg_speed_grad * time_elapsed)

        if predicted_direction > self.max_direction:
            predicted_direction = self.max_direction
        if predicted_speed > self.max_speed:
            predicted_speed = self.max_speed

        self.learn(predicted_direction)
        npd = self.beta * self.intention_angle + (1 - self.beta) * predicted_direction
        return predicted_speed, npd
    
    def predict_coords(self, coords, t):
        ps, pd = self.predict(t)
        return calculateNewCoords(coords, ps, pd, t)
    

if __name__ == '__main__':
    dypre = DynamicPredictor()
    dypre.feed(20, 1, 1)
    dypre.feed(21, 2, 2)
    dypre.feed(12, 3, 3)
    dypre.feed(10, 4, 4)
    dypre.feed(9, 5, 5)
    dypre.feed(8, 6, 10)
    dypre.feed(10, 10, 15)
    print(dypre.predict(1))
        


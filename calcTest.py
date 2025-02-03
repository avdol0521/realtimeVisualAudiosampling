import math
def calcDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calcAvg(val1, val2):
    return ((val1 + val2) / 2)

min_val, max_val = 0.1, 2.0 
fuckyou = 300
fuckyoutwo = 400

index_x_speed = 300
index_x_frequency = 300
index_y_speed = 
index_y_frequency = 0
thumb_x_speed = 0
thumb_x_frequency = 0
thumb_y_speed = 0
thumb_y_frequency = 0

volume_distance = calcAvg(calcDistance(index_x_speed, index_y_speed, index_x_frequency, index_y_frequency), calcDistance(thumb_x_speed, thumb_y_speed, thumb_x_frequency, thumb_y_frequency))
volume_value = round((min_val + (max_val - min_val) * min(max(volume_distance / fuckyoutwo, 0), 1)), 1)
print("volume_distance", volume_distance)
print("volume_value", volume_value)
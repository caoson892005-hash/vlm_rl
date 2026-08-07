from math import sqrt
def calculate_distance(a,b):
    x1, y1 = a
    x2, y2 = b
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)



a_coord=(1, 2)
b_coord=(4, 6)
distance = calculate_distance(a_coord, b_coord)
print(f"The distance between points {a_coord} and {b_coord} is: {distance}")

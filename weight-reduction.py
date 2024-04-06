import csv
import numpy as np

points = []

with open("thrust_data.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        points.append((float(row[0]), float(row[1])))

x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])

area = 0

area = np.sum(np.multiply(x, y))
weight_lost_per_unit_area = 0.008 / area

print(weight_lost_per_unit_area)
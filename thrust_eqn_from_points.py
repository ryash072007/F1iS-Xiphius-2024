import csv
import numpy as np
from matplotlib import pyplot as plt

points = []

with open("thrust_data.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        points.append((float(row[0]), float(row[1])))

x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])

coefficients = np.polyfit(x, y, 100)
polynomial = np.poly1d(coefficients)

x_curve = np.linspace(min(x), max(x), 40)
y_curve = polynomial(x_curve)
print(y_curve)
plt.scatter(x, y, label='Data Points')
plt.plot(x_curve, y_curve, label=f'Polynomial Curve', color='red')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Polynomial Curve Fitting')
plt.show()
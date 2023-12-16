import csv
import numpy as np
import matplotlib.pyplot as plt


points = []

with open("thrust_data.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        points.append((float(row[0]), float(row[1])))

x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])

coefficients = np.polyfit(x, y, 100)
polynomial = np.poly1d(coefficients)

def calculate_acceleration(thrust, drag, weight, friction):
    # Assuming thrust, drag, and weight are functions of time
    net_force = thrust - drag - friction
    acceleration = net_force / weight
    return acceleration

def calculate_velocity(acceleration, initial_velocity, time_step):
    velocity = initial_velocity + acceleration * time_step
    return velocity

def calculate_distance(velocity, initial_distance, time_step):
    distance = initial_distance + velocity * time_step
    return distance

def simulate_track_time(thrust_curve, drag_curve, weight, friction_curve, time_step, total_time):
    time_points = np.arange(0, total_time, time_step)

    # Initial conditions
    initial_velocity = 0
    initial_distance = 0

    velocities = [initial_velocity]
    distances = [initial_distance]

    for t in time_points[:-1]:
        thrust = thrust_curve(t)
        drag = drag_curve(velocities[-1])
        friction = friction_curve(t)

        acceleration = calculate_acceleration(thrust, drag, weight, friction)
        velocity = calculate_velocity(acceleration, velocities[-1], time_step)
        distance = calculate_distance(velocity, distances[-1], time_step)

        velocities.append(velocity)
        distances.append(distance)

    return time_points, distances, velocities

# Example friction function (placeholder)
def example_friction_curve(t):
    # Replace this with a real friction curve function
    return 0  # Placeholder value

def example_thrust_curve(t):
    return polynomial(t) if t < 0.5 else 0

def example_drag_curve(v):
    return 0.5 * 1.225 * v**2 * 0.05619 * 0.02531

car_weight = 0.05  # Placeholder value in kilograms

# Simulation parameters
time_step = 0.001  # Time step for simulation
total_time = 2  # Total simulation time

# Run simulation
time_points, distances, velocities = simulate_track_time(
    example_thrust_curve, example_drag_curve, car_weight, example_friction_curve, time_step, total_time
)

# Plot results
plt.plot(time_points, distances, label='Distance')
plt.plot(time_points, velocities, label='Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m) / Velocity (m/s)')
plt.legend()
plt.show()

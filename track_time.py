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
    net_force = thrust - drag - friction
    acceleration = net_force / weight
    return acceleration

def calculate_velocity(acceleration, initial_velocity, time_step):
    velocity = initial_velocity + acceleration * time_step
    return velocity

def calculate_distance(velocity, initial_distance, time_step):
    distance = initial_distance + velocity * time_step
    return distance if distance > 0 else 0

def simulate_track_time(thrust_curve, drag_curve, weight, friction_curve, time_step, total_time):
    time_points = np.arange(0, total_time, time_step)

    initial_velocity = 0
    initial_distance = 0
    initial_acceleration = 0

    velocities = [initial_velocity]
    distances = [initial_distance]
    accelerations = [initial_acceleration]

    time = 0
    
    for t in time_points[:-1]:
        thrust = thrust_curve(t)
        drag = drag_curve(velocities[-1])
        friction = friction_curve(t, thrust - drag)

        acceleration = calculate_acceleration(thrust, drag, weight, friction)
        velocity = calculate_velocity(acceleration, velocities[-1], time_step)
        distance = calculate_distance(velocity, distances[-1], time_step)

        velocities.append(velocity)
        distances.append(distance)
        accelerations.append(acceleration)
        
        if distance >= 20:
            time = t
            print(f"Distance has exceeded 20 meters at time: {t} seconds")
            print(f"Max velocity is {max(velocities)} at time {velocities.index(max(velocities)) * time_step}s")
            print(f"Average velocity is {distance / t}")
            print(f"Max acceleration is {max(accelerations)} at time {accelerations.index(max(accelerations)) * time_step}s")
            break
        

    return time_points, distances, velocities, accelerations, time

def friction_curve(t, total_force):
    friction = cof * car_weight * 9.81
    if total_force < 0:
        # print(f"At time {t}, friction is {friction}")
        return friction
    elif total_force < friction:
        # print(f"At time {t}, friction is {total_force}")
        return total_force
    elif total_force > friction:
        # print(f"At time {t}, friction is {friction}")
        return friction

def thrust_curve(t):
    return polynomial(t) if t < 0.5 else 0

def drag_curve(v):
    return 0.5 * 1.225 * v**2 * area * dcof


# Simulation parameters

ideal_car_weight = 0.05
co2_weight = 0.008
co2_cartridge_weight = 0.024


time_step = 0.01  # Time step for simulation
total_time = 2  # Total simulation time
car_weight = ideal_car_weight + co2_weight + co2_cartridge_weight  # Ideal Weight
cof = 0.492  # Coefficient of friction
area = 57331.3 / 1000**2 # Area of car in square meters
dcof = 0.02198 # Drag coefficient of car


time_points, distances, velocities, accelerations, time_taken = simulate_track_time(
        thrust_curve, drag_curve, car_weight, friction_curve, time_step, total_time
    )

plt.plot(time_points[:len(distances)], distances, label='Distance')
plt.plot(time_points[:len(velocities)], velocities, label='Velocity')
plt.plot(time_points[:len(accelerations)], accelerations, label='Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m) / Velocity (m/s) / Acceleration (m/s^2)')
plt.legend()
plt.show()
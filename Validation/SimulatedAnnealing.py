import numpy as np
import random
import matplotlib.pyplot as plt
import time

depot = (-8.8, 0)
customer_coords1 = [(-4.8 + 0.6 * i, -1.5 + 3 * j) for i in range(17) for j in range(1)]
customer_coords2 = [(-4.8 + 0.6 * i, 1.5 + 3 * j) for i in range(17) for j in range(1)]
customer_coords3 = [(-4.8 + 0.6 * i, -0.5 + 1 * j) for i in range(17) for j in range(1)]
customer_coords4 = [(-4.2 + 0.6 * i, -0.5 + 1 * j) for i in range(15) for j in range(1)]
customer_coords5 = [(-4.8 + 0.6 * i, 0.5 + 1 * j) for i in range(17) for j in range(1)]
customer_coords6 = [(-4.2 + 0.6 * i, 0.5 + 1 * j) for i in range(15) for j in range(1)]

locations = [depot] + customer_coords1 + customer_coords2 + customer_coords3 + customer_coords4 + customer_coords5 + customer_coords6
demands = [int(0.2 * 1000)] * len(customer_coords1) + \
          [int(0.2 * 1000)] * len(customer_coords2) + \
          [int(0.1 * 1000)] * len(customer_coords3) + \
          [int(0.1 * 1000)] * len(customer_coords4) + \
          [int(0.1 * 1000)] * len(customer_coords5) + \
          [int(0.1 * 1000)] * len(customer_coords6)

vehicle_capacities = 1000
num_vehicles = 15
customer_ids = list(range(1, len(locations)))

# ========== Distance Matrix ==========
def compute_distance_matrix(locations):
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
    return dist_matrix

distance_matrix = compute_distance_matrix(locations)

def decode_solution(solution):
    routes = []
    route = []
    load = 0
    for customer in solution:
        demand = demands[customer - 1]
        if load + demand <= vehicle_capacities:
            route.append(customer)
            load += demand
        else:
            routes.append([0] + route + [0])
            route = [customer]
            load = demand
    if route:
        routes.append([0] + route + [0])
    if len(routes) > num_vehicles:
        raise ValueError("Too many vehicles needed")
    return routes

def evaluate_solution(solution):
    try:
        routes = decode_solution(solution)
    except ValueError:
        return float('inf')
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance

def get_neighbor(solution):
    a, b = random.sample(range(len(solution)), 2)
    neighbor = solution[:]
    neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
    return neighbor

# ========== SA ==========
def simulated_annealing(init_temp=5000, cooling_rate=0.9999	, min_temp=1e-5, max_iter=500000 ):
    current_solution = random.sample(customer_ids, len(customer_ids))
    current_score = evaluate_solution(current_solution)
    best_solution = current_solution[:]
    best_score = current_score

    temp = init_temp
    iteration = 0
    start_time = time.time()

    while temp > min_temp and iteration < max_iter:
        neighbor = get_neighbor(current_solution)
        neighbor_score = evaluate_solution(neighbor)

        if neighbor_score < current_score or random.random() < np.exp((current_score - neighbor_score) / temp):
            current_solution = neighbor
            current_score = neighbor_score
            if neighbor_score < best_score:
                best_solution = neighbor
                best_score = neighbor_score

        if iteration % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iteration}: Best Distance = {best_score:.3f}, Temp = {temp:.2f}, Time Elapsed: {elapsed:.2f}s")

        temp *= cooling_rate
        iteration += 1

    return best_solution, best_score

# ========== Plot ==========
def plot_routes(routes, locations):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    locs = np.array(locations)
    plt.figure(figsize=(12, 6))
    plt.scatter(locs[1:, 0], locs[1:, 1], c='black', label='Customers')
    plt.scatter(locs[0, 0], locs[0, 1], c='orange', marker='*', s=200, label='Depot')

    for i, route in enumerate(routes):
        coords = np.array([locations[idx] for idx in route])
        plt.plot(coords[:, 0], coords[:, 1], '-o', color=colors[i % len(colors)], label=f'Vehicle {i}')

    plt.title("Simulated Annealing VRP Solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

# Run SA
best_solution, best_score = simulated_annealing()
routes = decode_solution(best_solution)
print("\nBest Total Distance:", best_score)
for i, route in enumerate(routes):
    print(f"Vehicle {i}: {route}")

plot_routes(routes, locations)

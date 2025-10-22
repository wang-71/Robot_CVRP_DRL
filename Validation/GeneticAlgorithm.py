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

locations = [
                depot] + customer_coords1 + customer_coords2 + customer_coords3 + customer_coords4 + customer_coords5 + customer_coords6
demands = [int(0.2 * 1000)] * len(customer_coords1) + \
          [int(0.2 * 1000)] * len(customer_coords2) + \
          [int(0.1 * 1000)] * len(customer_coords3) + \
          [int(0.1 * 1000)] * len(customer_coords4) + \
          [int(0.1 * 1000)] * len(customer_coords5) + \
          [int(0.1 * 1000)] * len(customer_coords6)

vehicle_capacities = 1000

num_vehicles = 15
customer_ids = list(range(1, len(locations)))

# GA 参数
pop_size = 100
generations = 10000
mutation_rate = 0.1

def compute_distance_matrix(locations):
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
    return dist_matrix

distance_matrix = compute_distance_matrix(locations)


def decode_solution(solution):
    if len(set(solution)) != len(solution) or set(solution) != set(customer_ids):
        raise ValueError("Repeated customers")

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
        raise ValueError("Exceeding total number of vehicles")
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

# ========== GA ==========
def initialize_population():
    return [random.sample(customer_ids, len(customer_ids)) for _ in range(pop_size)]

def tournament_selection(pop, scores, k=3):
    selected = random.sample(list(zip(pop, scores)), k)
    return min(selected, key=lambda x: x[1])[0]

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[p1:p2] = parent1[p1:p2]

    for i in range(p1, p2):
        if parent2[i] not in child:
            val = parent2[i]
            idx = i
            while True:
                val = parent1[idx]
                if val not in child:
                    idx = parent2.index(val)
                else:
                    break
            child[idx] = parent2[i]

    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child

def mutate(solution):
    a, b = random.sample(range(len(solution)), 2)
    solution[a], solution[b] = solution[b], solution[a]

population = initialize_population()
best_solution = None
best_score = float('inf')
start_time = time.time()

for gen in range(generations):
    scores = [evaluate_solution(sol) for sol in population]
    new_population = []

    for _ in range(pop_size):
        parent1 = tournament_selection(population, scores)
        parent2 = tournament_selection(population, scores)
        child = pmx_crossover(parent1, parent2)
        if random.random() < mutation_rate:
            mutate(child)
        if len(set(child)) == len(child):
            new_population.append(child)

    population = new_population
    gen_best = min(population, key=evaluate_solution)
    gen_best_score = evaluate_solution(gen_best)

    if gen_best_score < best_score:
        best_score = gen_best_score
        best_solution = gen_best

    if gen % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Generation {gen}: Best Distance = {best_score:.3f}, Time Elapsed: {elapsed:.2f}s")

# ========== Plot ==========
routes = decode_solution(best_solution)
print("\nBest Total Distance:", best_score)
for i, route in enumerate(routes):
    print(f"Vehicle {i}: {route}")

def plot_routes(routes, locations):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    locs = np.array(locations)
    plt.figure(figsize=(12, 6))
    plt.scatter(locs[1:, 0], locs[1:, 1], c='black', label='Customers')
    plt.scatter(locs[0, 0], locs[0, 1], c='orange', marker='*', s=200, label='Depot')

    for i, route in enumerate(routes):
        coords = np.array([locations[idx] for idx in route])
        color = colors[i % len(colors)]
        plt.plot(coords[:, 0], coords[:, 1], '-o', color=color, label=f'Vehicle {i}')

    plt.title("Genetic Algorithm VRP Solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_routes(routes, locations)

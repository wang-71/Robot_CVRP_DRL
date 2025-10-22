import numpy as np
import matplotlib.pyplot as plt


def compute_euclidean_distance_matrix(locations):
    size = len(locations)
    dist_matrix = np.zeros((size, size))
    for from_node in range(size):
        for to_node in range(size):
            if from_node != to_node:
                dist_matrix[from_node][to_node] = np.linalg.norm(
                    np.array(locations[from_node]) - np.array(locations[to_node])
                )
    return dist_matrix


def nearest_neighbor_vrp(locations, demands, vehicle_capacities):
    n_customers = len(locations) - 1
    distance_matrix = compute_euclidean_distance_matrix(locations)
    unvisited = set(range(1, n_customers + 1))
    depot_index = 0
    routes = []

    for cap in vehicle_capacities:
        route = [depot_index]
        current_node = depot_index
        load = 0

        while True:
            nearest = None
            nearest_dist = float('inf')
            for customer in unvisited:
                demand = demands[customer - 1]
                if load + demand <= cap:
                    dist = distance_matrix[current_node][customer]
                    if dist < nearest_dist:
                        nearest = customer
                        nearest_dist = dist

            if nearest is None:
                break

            route.append(nearest)
            load += demands[nearest - 1]
            unvisited.remove(nearest)
            current_node = nearest

        route.append(depot_index)
        routes.append(route)

        if not unvisited:
            break

    return routes, distance_matrix


def plot_routes(routes, locations):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    locs = np.array(locations)

    plt.figure(figsize=(12, 6))
    plt.scatter(locs[1:, 0], locs[1:, 1], c='black', label='Customers')
    plt.scatter(locs[0, 0], locs[0, 1], c='orange', marker='*', s=200, label='Depot')

    for i, route in enumerate(routes):
        route_coords = locs[route]
        color = colors[i % len(colors)]
        plt.plot(route_coords[:, 0], route_coords[:, 1], '-o', color=color, label=f'Vehicle {i}')

    plt.title("Nearest Neighbor VRP Solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    depot = (-8.8, 0)
    customer_coords1 = [(-4.8 + 0.6 * i, -1.5 + 3 * j) for i in range(17) for j in range(1)]
    customer_coords2 = [(-4.8 + 0.6 * i, 1.5 + 3 * j) for i in range(17) for j in range(1)]
    customer_coords3 = [(-4.8 + 0.6 * i, -0.5 + 1 * j) for i in range(17) for j in range(1)]
    customer_coords4 = [(-4.2 + 0.6 * i, -0.5 + 1 * j) for i in range(15) for j in range(1)]
    customer_coords5 = [(-4.8 + 0.6 * i, 0.5 + 1 * j) for i in range(17) for j in range(1)]
    customer_coords6 = [(-4.2 + 0.6 * i, 0.5 + 1 * j) for i in range(15) for j in range(1)]

    locations = [depot] + customer_coords1 + customer_coords2 + customer_coords3 +customer_coords4 +customer_coords5 +customer_coords6
    demands = [int(0.2 * 1000)] * len(customer_coords1) + \
              [int(0.2 * 1000)] * len(customer_coords2) + \
              [int(0.1 * 1000)] * len(customer_coords3) + \
              [int(0.1 * 1000)] * len(customer_coords4) + \
              [int(0.1 * 1000)] * len(customer_coords5) + \
              [int(0.1 * 1000)] * len(customer_coords6)

    vehicle_capacities = [int(1.0 * 1000)] * 15

    routes, dist_matrix = nearest_neighbor_vrp(locations, demands, vehicle_capacities)

    total_distance = 0
    for i, route in enumerate(routes):
        dist = 0
        for j in range(len(route) - 1):
            dist += dist_matrix[route[j]][route[j + 1]]
        total_distance += dist
        print(f"Vehicle {i}: {route}, Distance: {dist:.3f}")

    print(f"\nTotal Distance: {total_distance:.3f}")

    plot_routes(routes, locations)


if __name__ == '__main__':
    main()

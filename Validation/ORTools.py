from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import time


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


def main():
    start_time = time.time()

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

    vehicle_capacities = [int(1.0 * 1000)] * 15
    num_vehicles = len(vehicle_capacities)
    depot_index = 0

    distance_matrix = compute_euclidean_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node - 1] if from_node != 0 else 0

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, vehicle_capacities, True, "Capacity")

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        total_distance = 0
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        plt.figure(figsize=(10, 6))

        # 绘制所有客户点
        locs = np.array(locations)
        plt.scatter(locs[1:, 0], locs[1:, 1], c='black', label='Customers')
        plt.scatter(locs[0, 0], locs[0, 1], c='orange', marker='*', s=200, label='Depot')

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route_distance = 0
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            route.append(manager.IndexToNode(index))
            if len(route) > 2:
                print(f"Vehicle {vehicle_id}'s route: {route}")
                print(f"Distance: {route_distance / 1000:.3f}")

                # 绘图：画出路径
                route_coords = np.array([locations[i] for i in route])
                color = color_list[vehicle_id % len(color_list)]
                plt.plot(route_coords[:, 0], route_coords[:, 1], '-o', color=color, label=f'Vehicle {vehicle_id}')

            total_distance += route_distance

        print(f"\nTotal Distance: {total_distance / 1000:.3f}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Process Time: {elapsed_time:.2f} seconds")

        plt.title("Vehicle Routes")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No solution found!")


if __name__ == '__main__':
    main()

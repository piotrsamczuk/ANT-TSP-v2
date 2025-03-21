import numpy as np
import matplotlib.pyplot as plt
import random
import time

NUM_ANTS = 20
NUM_ITERATIONS = 200
EVAPORATION_RATE = 0.5
ALPHA = 1
BETA = 2
Q = 100
Q0 = 0.9

def generate_distance_matrix(num_cities):
    distances = np.random.randint(1, 100, size=(num_cities, num_cities))
    distances = (distances + distances.T) // 2
    np.fill_diagonal(distances, 0)
    return distances

def ant_colony_optimization(distances, num_ants, num_iterations, evaporation_rate, alpha, beta, q, approach):
    num_cities = len(distances)
    
    if approach["pheromone_init"] == "uniform":
        pheromones = np.ones((num_cities, num_cities))
    elif approach["pheromone_init"] == "heuristic":
        pheromones = initialize_pheromones(distances)

    best_path = None
    best_distance = float('inf')
    best_distances_history = []

    for iteration in range(num_iterations):
        paths = []
        path_distances = []

        for ant in range(num_ants):
            visited = [False] * num_cities
            current_city = random.randint(0, num_cities - 1)
            visited[current_city] = True
            path = [current_city]
            path_distance = 0

            while len(path) < num_cities:
                if approach["city_selection"] == "probabilistic":
                    next_city = select_next_city_probabilistic(current_city, visited, pheromones, distances, alpha, beta)
                elif approach["city_selection"] == "mixed":
                    next_city = select_next_city_mixed(current_city, visited, pheromones, distances, alpha, beta, Q0)
                path.append(next_city)
                path_distance += distances[current_city][next_city]
                visited[next_city] = True
                current_city = next_city

            path_distance += distances[current_city][path[0]]
            paths.append(path)
            path_distances.append(path_distance)

            if path_distance < best_distance:
                best_distance = path_distance
                best_path = path

        if approach["pheromone_update"] == "global":
            update_pheromones_global(pheromones, paths, path_distances, evaporation_rate, q)
        elif approach["pheromone_update"] == "elitist":
            update_pheromones_elitist(pheromones, paths, path_distances, evaporation_rate, q, best_path)

        if approach["evaporation"] == "dynamic":
            dynamic_evaporation = evaporation_rate * (1 - iteration / num_iterations)
            pheromones *= (1 - dynamic_evaporation)
        elif approach["evaporation"] == "static":
            pheromones *= (1 - evaporation_rate)

        best_distances_history.append(best_distance)

    return best_path, best_distance, best_distances_history

def initialize_pheromones(distances):
    num_cities = len(distances)
    pheromones = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                pheromones[i][j] = 1 / distances[i][j]
    return pheromones

def select_next_city_probabilistic(current_city, visited, pheromones, distances, alpha, beta):
    num_cities = len(distances)
    probabilities = []

    for city in range(num_cities):
        if not visited[city]:
            pheromone = pheromones[current_city][city] ** alpha
            heuristic = (1 / distances[current_city][city]) ** beta
            probabilities.append(pheromone * heuristic)
        else:
            probabilities.append(0)

    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return np.random.choice(range(num_cities), p=probabilities)

def select_next_city_mixed(current_city, visited, pheromones, distances, alpha, beta, q0):
    num_cities = len(distances)
    if random.random() < q0:
        best_city = -1
        best_value = -1
        for city in range(num_cities):
            if not visited[city]:
                value = pheromones[current_city][city] ** alpha * (1 / distances[current_city][city]) ** beta
                if value > best_value:
                    best_value = value
                    best_city = city
        return best_city
    else:
        return select_next_city_probabilistic(current_city, visited, pheromones, distances, alpha, beta)

def update_pheromones_global(pheromones, paths, path_distances, evaporation_rate, q):
    pheromones *= (1 - evaporation_rate)
    best_path = paths[np.argmin(path_distances)]
    best_distance = min(path_distances)
    for i in range(len(best_path) - 1):
        city_a, city_b = best_path[i], best_path[i + 1]
        pheromones[city_a][city_b] += q / best_distance
        pheromones[city_b][city_a] += q / best_distance

def update_pheromones_elitist(pheromones, paths, path_distances, evaporation_rate, q, best_path):
    pheromones *= (1 - evaporation_rate)
    best_distance = min(path_distances)
    for path, distance in zip(paths, path_distances):
        for i in range(len(path) - 1):
            city_a, city_b = path[i], path[i + 1]
            pheromones[city_a][city_b] += q / distance
            pheromones[city_b][city_a] += q / distance
    for i in range(len(best_path) - 1):
        city_a, city_b = best_path[i], best_path[i + 1]
        pheromones[city_a][city_b] += q / best_distance
        pheromones[city_b][city_a] += q / best_distance

def compare_results(results, labels, execution_times):
    plt.figure(figsize=(12, 8))
    for result, label in zip(results, labels):
        plt.plot(result, label=label, linewidth=2)
    plt.xlabel("Iteracje", fontsize=14)
    plt.ylabel("Najlepsza odległość", fontsize=14)
    plt.title("Porównanie wyników różnych kombinacji", fontsize=16, pad=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple']
    bars = plt.bar(range(len(labels)), execution_times, color=colors)
    plt.xticks([])
    plt.legend(bars, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
    plt.xlabel("Kombinacje", fontsize=14)
    plt.ylabel("Czas wykonania (ms)", fontsize=14)
    plt.title("Porównanie czasu wykonania różnych kombinacji", fontsize=16, pad=20)
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    num_cities = 50
    distances = generate_distance_matrix(num_cities)

    approaches = [
        {
            "name": "Probabilistyczny + Globalna + Jednorodna + Stałe parowanie",
            "city_selection": "probabilistic",
            "pheromone_update": "global",
            "pheromone_init": "uniform",
            "evaporation": "static"
        },
        {
            "name": "Mieszany + Elitarna + Heurystyczna + Dynamiczne parowanie",
            "city_selection": "mixed",
            "pheromone_update": "elitist",
            "pheromone_init": "heuristic",
            "evaporation": "dynamic"
        },
        {
            "name": "Probabilistyczny + Elitarna + Heurystyczna + Stałe parowanie",
            "city_selection": "probabilistic",
            "pheromone_update": "elitist",
            "pheromone_init": "heuristic",
            "evaporation": "static"
        },
        {
            "name": "Mieszany + Globalna + Jednorodna + Dynamiczne parowanie",
            "city_selection": "mixed",
            "pheromone_update": "global",
            "pheromone_init": "uniform",
            "evaporation": "dynamic"
        }
    ]

    results = []
    labels = []
    execution_times = []

    for approach in approaches:
        start_time = time.time()
        best_path, best_distance, history = ant_colony_optimization(distances, NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE, ALPHA, BETA, Q, approach)
        execution_time = (time.time() - start_time) * 1000
        results.append(history)
        labels.append(approach["name"])
        execution_times.append(execution_time)

        best_path = [int(city) for city in best_path]

        print(f"Kombinacja: {approach['name']}")
        print(f"Najlepsza znaleziona odległość: {best_distance}")
        print(f"Czas wykonania: {execution_time:.2f} ms")
        print(f"Najlepsza ścieżka: {best_path}")
        print("-" * 50)

    compare_results(results, labels, execution_times)

if __name__ == "__main__":
    main()
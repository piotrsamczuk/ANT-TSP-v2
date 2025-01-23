import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Parametry algorytmu
NUM_ANTS = 10  # Liczba mrówek
NUM_ITERATIONS = 100  # Liczba iteracji
EVAPORATION_RATE = 0.5  # Współczynnik parowania feromonów
ALPHA = 1  # Waga feromonów
BETA = 2  # Waga heurystyki (odwrotność odległości)
Q = 100  # Stała feromonowa

# Generowanie losowej macierzy odległości między miastami
def generate_distance_matrix(num_cities):
    distances = np.random.randint(1, 100, size=(num_cities, num_cities))
    distances = (distances + distances.T) // 2  # Symetryczna macierz odległości
    np.fill_diagonal(distances, 0)  # Zerowanie przekątnej
    return distances

# Implementacja Algorytmu Mrówkowego
def ant_colony_optimization(distances, num_ants, num_iterations, evaporation_rate, alpha, beta, q):
    num_cities = len(distances)
    pheromones = np.ones((num_cities, num_cities))  # Początkowa ilość feromonów

    best_path = None
    best_distance = float('inf')

    # Historia najlepszych rozwiązań
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

            # Budowanie ścieżki przez mrówkę
            while len(path) < num_cities:
                next_city = select_next_city(current_city, visited, pheromones, distances, alpha, beta)
                path.append(next_city)
                path_distance += distances[current_city][next_city]
                visited[next_city] = True
                current_city = next_city

            # Powrót do miasta początkowego
            path_distance += distances[current_city][path[0]]
            paths.append(path)
            path_distances.append(path_distance)

            # Aktualizacja najlepszej ścieżki
            if path_distance < best_distance:
                best_distance = path_distance
                best_path = path

        # Aktualizacja feromonów
        update_pheromones(pheromones, paths, path_distances, evaporation_rate, q)

        # Zapisywanie historii najlepszych rozwiązań
        best_distances_history.append(best_distance)

    return best_path, best_distance, best_distances_history

# Wybór następnego miasta przez mrówkę
def select_next_city(current_city, visited, pheromones, distances, alpha, beta):
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

    next_city = np.random.choice(range(num_cities), p=probabilities)
    return next_city

# Aktualizacja feromonów
def update_pheromones(pheromones, paths, path_distances, evaporation_rate, q):
    pheromones *= (1 - evaporation_rate)  # Parowanie feromonów

    for path, distance in zip(paths, path_distances):
        for i in range(len(path) - 1):
            city_a, city_b = path[i], path[i + 1]
            pheromones[city_a][city_b] += q / distance
            pheromones[city_b][city_a] += q / distance

# Funkcja do porównywania wyników
def compare_results(results, labels, execution_times):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Wykres najlepszych odległości
    for result, label in zip(results, labels):
        ax1.plot(result, label=label)
    ax1.set_xlabel("Iteracje")
    ax1.set_ylabel("Najlepsza odległość")
    ax1.set_title("Porównanie wyników różnych implementacji/parametrów")
    ax1.legend()
    ax1.grid()

    # Wykres czasu wykonania
    ax2.bar(labels, execution_times, color=['blue', 'green', 'red', 'purple'])
    ax2.set_xlabel("Warianty")
    ax2.set_ylabel("Czas wykonania (ms)")
    ax2.set_title("Porównanie czasu wykonania różnych wariantów")
    ax2.grid()

    plt.show()

# Główna funkcja
def main():
    num_cities = 20  # Liczba miast
    distances = generate_distance_matrix(num_cities)

    # Testowanie różnych parametrów
    results = []
    labels = []
    execution_times = []

    # Wariant 1: Domyślne parametry
    start_time = time.time()
    _, _, history_default = ant_colony_optimization(distances, NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE, ALPHA, BETA, Q)
    execution_time = (time.time() - start_time) * 1000  # Czas w ms
    results.append(history_default)
    labels.append("Domyślne parametry")
    execution_times.append(execution_time)

    # Wariant 2: Większa liczba mrówek
    start_time = time.time()
    _, _, history_more_ants = ant_colony_optimization(distances, NUM_ANTS * 2, NUM_ITERATIONS, EVAPORATION_RATE, ALPHA, BETA, Q)
    execution_time = (time.time() - start_time) * 1000  # Czas w ms
    results.append(history_more_ants)
    labels.append("Większa liczba mrówek")
    execution_times.append(execution_time)

    # Wariant 3: Mniejszy współczynnik parowania feromonów
    start_time = time.time()
    _, _, history_lower_evaporation = ant_colony_optimization(distances, NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE * 0.5, ALPHA, BETA, Q)
    execution_time = (time.time() - start_time) * 1000  # Czas w ms
    results.append(history_lower_evaporation)
    labels.append("Mniejszy współczynnik parowania")
    execution_times.append(execution_time)

    # Wariant 4: Większa waga heurystyki (BETA)
    start_time = time.time()
    _, _, history_higher_beta = ant_colony_optimization(distances, NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE, ALPHA, BETA * 2, Q)
    execution_time = (time.time() - start_time) * 1000  # Czas w ms
    results.append(history_higher_beta)
    labels.append("Większa waga heurystyki")
    execution_times.append(execution_time)

    # Porównanie wyników na wykresie
    compare_results(results, labels, execution_times)

if __name__ == "__main__":
    main()
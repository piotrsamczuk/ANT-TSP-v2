# Ant Colony Optimization for TSP

This project implements the Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesman Problem (TSP). The implementation allows for comparing different ACO approaches by combining various strategies for city selection, pheromone updates, pheromone initialization, and evaporation methods.

## Overview

The Traveling Salesman Problem is a classic optimization challenge where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city. This project uses Ant Colony Optimization, a nature-inspired algorithm that mimics how ants find optimal paths using pheromone trails.

## Features

- **Flexible ACO Implementation**: Combines different strategies for a comprehensive analysis
- **City Selection Strategies**:
  - Probabilistic: Selection based on pheromone and heuristic information
  - Mixed: Combines exploitation (best city) and exploration (probabilistic selection)
- **Pheromone Update Methods**:
  - Global: Updates based on the best path in the current iteration
  - Elitist: Updates based on all paths and gives extra weight to the best path found so far
- **Pheromone Initialization**:
  - Uniform: All edges start with equal pheromone levels
  - Heuristic: Initial pheromone levels inversely proportional to distances
- **Evaporation Approaches**:
  - Static: Constant evaporation rate throughout all iterations
  - Dynamic: Evaporation rate decreases over time to encourage convergence

## Results Visualization

The implementation includes visualization tools to compare different approach combinations:
- Performance over iterations (convergence plots)
- Execution time comparison
- Best path visualization

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Usage

Install the requirements

```bash
pip install -r requirements.txt
```

Run the main script to execute the comparison of different ACO approaches:

```bash
python main.py
```

## Parameters

The algorithm's behavior can be adjusted by modifying these parameters in the code:
- `NUM_ANTS`: Number of ants in the colony
- `NUM_ITERATIONS`: Maximum number of iterations
- `EVAPORATION_RATE`: Base rate for pheromone evaporation
- `ALPHA`: Weight of pheromone information
- `BETA`: Weight of heuristic information (distance)
- `Q`: Constant used for pheromone deposit
- `Q0`: Probability threshold for exploitation in mixed selection

## Example Output

The program outputs:
- Best distance found for each approach
- Execution time in milliseconds
- Best path (sequence of cities)
- Comparative graphs showing convergence and execution times

## Future Improvements

Potential enhancements for this project:
- Implement more ACO variants (MAX-MIN Ant System, Ant Colony System)
- Add parallelization to improve performance
- Include more problem instances and benchmarks
- Create interactive visualization of ant movement and pheromone trails

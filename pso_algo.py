import numpy as np
import pandas as pd
import time
from rastrigin import rastrigin

def run_pso(pop_size=50, generations=100, w=0.7, c1=1.5, c2=1.5,
            domain=(-5.12, 5.12), return_result=False, seed=42):
    """
    Particle Swarm Optimization (PSO) for minimizing the Rastrigin function.
    Each particle adjusts its velocity based on personal and global best positions.
    """

    # Set seed for reproducibility
    np.random.seed(seed)

    start = time.time()

    # Initialize particle positions randomly within the domain
    pos = np.random.uniform(*domain, size=(pop_size, 2))
    vel = np.zeros_like(pos)  # Initialize velocity to zero

    pbest = pos.copy()  # Personal best positions
    pbest_val = np.array([rastrigin(ind) for ind in pos])  # Fitness values
    gbest = pbest[np.argmin(pbest_val)]  # Global best position
    best_fitness = []

    for gen in range(generations):
        for i in range(pop_size):
            r1, r2 = np.random.rand(2)  # Random coefficients

            # Update velocity based on inertia, cognitive, and social components
            vel[i] = (w * vel[i] +
                      c1 * r1 * (pbest[i] - pos[i]) +
                      c2 * r2 * (gbest - pos[i]))

            # Update position and clip within bounds
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], *domain)

            # Update personal best if improved
            val = rastrigin(pos[i])
            if val < pbest_val[i]:
                pbest[i] = pos[i]
                pbest_val[i] = val

        # Update global best
        gbest = pbest[np.argmin(pbest_val)]
        best_fitness.append(rastrigin(gbest))

    best_val = rastrigin(gbest)
    elapsed = time.time() - start

    if not return_result:
        pd.DataFrame(best_fitness, columns=['PSO_Best_Fitness']).to_csv('results/pso_results.csv', index=False)
        pd.DataFrame([[*gbest, best_val, elapsed]], columns=['x', 'y', 'fitness', 'time']).to_csv('results/pso_best.csv', index=False)

    if return_result:
        return best_val, gbest[0], gbest[1], elapsed

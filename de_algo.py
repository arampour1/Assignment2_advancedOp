import numpy as np
import pandas as pd
import time
from rastrigin import rastrigin

def run_de(pop_size=50, generations=100, F=0.5, CR=0.7,
           domain=(-5.12, 5.12), return_result=False, seed=42):
    """
    Differential Evolution for minimizing the Rastrigin function.
    Uses mutation, crossover, and selection operations on continuous values.
    """

    # Set seed for reproducibility
    np.random.seed(seed)

    start = time.time()

    # Initialize a population of 2D points in the given domain
    pop = np.random.uniform(*domain, size=(pop_size, 2))
    best_fitness = []

    for gen in range(generations):
        next_gen = []

        for i in range(pop_size):
            # Randomly select 3 vectors distinct from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            # Mutation: generate a mutant vector
            mutant = np.clip(a + F * (b - c), *domain)

            # Crossover: determine which components to use from mutant
            cross_points = np.random.rand(2) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, 2)] = True

            # Trial vector
            trial = np.where(cross_points, mutant, pop[i])

            # Selection: choose between trial and original
            if rastrigin(trial) < rastrigin(pop[i]):
                next_gen.append(trial)
            else:
                next_gen.append(pop[i])

        pop = np.array(next_gen)
        best = min([rastrigin(ind) for ind in pop])
        best_fitness.append(best)

    best_ind = min(pop, key=rastrigin)
    best_val = rastrigin(best_ind)
    elapsed = time.time() - start

    if not return_result:
        pd.DataFrame(best_fitness, columns=['DE_Best_Fitness']).to_csv('results/de_results.csv', index=False)
        pd.DataFrame([[*best_ind, best_val, elapsed]], columns=['x', 'y', 'fitness', 'time']).to_csv('results/de_best.csv', index=False)

    if return_result:
        return best_val, best_ind[0], best_ind[1], elapsed

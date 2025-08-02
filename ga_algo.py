import numpy as np
import pandas as pd
import random
import time
from rastrigin import rastrigin

def run_ga(pop_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8,
           domain=(-5.12, 5.12), return_result=False, seed=42):
    """
    Genetic Algorithm for minimizing the Rastrigin function.
    Supports parameterized runs for sensitivity analysis.
    """

    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    start = time.time()

    def init_population():
        # Initialize population of random 2D individuals within the given domain
        return [np.random.uniform(*domain, 2) for _ in range(pop_size)]

    def select(pop):
        # Select the top 50% fittest individuals (lower Rastrigin value is better)
        return sorted(pop, key=rastrigin)[:pop_size // 2]

    def crossover(p1, p2):
        # Perform uniform crossover between two parents with a random alpha
        if random.random() < crossover_rate:
            alpha = random.random()
            return alpha * p1 + (1 - alpha) * p2
        return p1  # No crossover, return parent

    def mutate(ind):
        # Mutate the individual using Gaussian noise, then clip to domain
        if random.random() < mutation_rate:
            ind += np.random.normal(0, 0.3, size=2)
        return np.clip(ind, *domain)

    population = init_population()
    best_fitness = []

    for gen in range(generations):
        selected = select(population)
        offspring = []

        # Generate new population using crossover and mutation
        while len(offspring) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            offspring.append(child)

        population = offspring
        best = min([rastrigin(ind) for ind in population])
        best_fitness.append(best)

    # Get the best individual in the final population
    best_ind = min(population, key=rastrigin)
    best_val = rastrigin(best_ind)
    elapsed = time.time() - start

    if not return_result:
        pd.DataFrame(best_fitness, columns=['GA_Best_Fitness']).to_csv('results/ga_results.csv', index=False)
        pd.DataFrame([[*best_ind, best_val, elapsed]], columns=['x', 'y', 'fitness', 'time']).to_csv('results/ga_best.csv', index=False)

    if return_result:
        return best_val, best_ind[0], best_ind[1], elapsed

import time
import pandas as pd
from rastrigin import rastrigin
from ga_algo import run_ga
from de_algo import run_de
from pso_algo import run_pso

# Test GA with multiple mutation rates and population sizes
def run_ga_sensitivity(mutation_rates, pop_sizes):
    results = []
    for pop in pop_sizes:
        for rate in mutation_rates:
            best, x, y, t = run_ga(mutation_rate=rate, pop_size=pop, return_result=True)
            results.append({
                'Algorithm': 'GA',
                'Mutation Rate': rate,
                'Population Size': pop,
                'x': x,
                'y': y,
                'Fitness': best,
                'Time (s)': t
            })
    pd.DataFrame(results).to_csv('results/ga_sensitivity_full.csv', index=False)

# Test DE with multiple combinations of F and CR
def run_de_sensitivity(Fs, CRs):
    results = []
    for f in Fs:
        for cr in CRs:
            best, x, y, t = run_de(F=f, CR=cr, return_result=True)
            results.append({
                'Algorithm': 'DE',
                'Differential Weight (F)': f,
                'Crossover Rate (CR)': cr,
                'x': x,
                'y': y,
                'Fitness': best,
                'Time (s)': t
            })
    pd.DataFrame(results).to_csv('results/de_sensitivity_full.csv', index=False)

# Test PSO with multiple inertia weights and population sizes
def run_pso_sensitivity(ws, pop_sizes):
    results = []
    for pop in pop_sizes:
        for w in ws:
            best, x, y, t = run_pso(w=w, pop_size=pop, return_result=True)
            results.append({
                'Algorithm': 'PSO',
                'Inertia Weight (w)': w,
                'Population Size': pop,
                'x': x,
                'y': y,
                'Fitness': best,
                'Time (s)': t
            })
    pd.DataFrame(results).to_csv('results/pso_sensitivity_full.csv', index=False)

# Run all tests for different parameter combinations
if __name__ == '__main__':
    run_ga_sensitivity(mutation_rates=[0.01, 0.05, 0.1, 0.2, 0.3], pop_sizes=[20, 30, 50, 100, 150])
    run_de_sensitivity(Fs=[0.3, 0.4, 0.5, 0.6, 0.8], CRs=[0.3, 0.5, 0.7, 0.9, 1.0])
    run_pso_sensitivity(ws=[0.3, 0.5, 0.7, 0.9, 1.2], pop_sizes=[20, 30, 50, 100, 150])

from ga_algo import run_ga
from de_algo import run_de
from pso_algo import run_pso
from plot_results import plot_results

if __name__ == '__main__':
    """
    Main execution script using the best configurations for GA, DE, and PSO
    as determined by the sensitivity analysis. Outputs:
    - Fitness over generations
    - Final best solution coordinates
    - Summary comparison and convergence plot
    """

    print("Running Genetic Algorithm with best config (mutation_rate=0.05, pop_size=150)...")
    run_ga(mutation_rate=0.05, pop_size=150)

    print("Running Differential Evolution with best config (F=0.3, CR=0.3)...")
    run_de(F=0.3, CR=0.3)

    print("Running Particle Swarm Optimization with best config (w=0.5, pop_size=20)...")
    run_pso(w=0.5, pop_size=20)

    print("All algorithms finished. Plotting results and generating summary.")
    plot_results()

import pandas as pd
import matplotlib.pyplot as plt

def plot_results():
    """
    Reads result CSVs from GA, DE, and PSO, plots their convergence over generations,
    and writes a summary CSV comparing final best fitness and time for each algorithm.
    """

    # Load fitness over generations for all algorithms
    df_ga = pd.read_csv('results/ga_results.csv')
    df_de = pd.read_csv('results/de_results.csv')
    df_pso = pd.read_csv('results/pso_results.csv')

    # Plot each algorithm's convergence
    plt.plot(df_ga, label='GA')
    plt.plot(df_de, label='DE')
    plt.plot(df_pso, label='PSO')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Convergence of GA, DE, and PSO on Rastrigin')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/convergence_plot.png')
    plt.show()

    # Load final best result for each algorithm
    ga = pd.read_csv('results/ga_best.csv')
    de = pd.read_csv('results/de_best.csv')
    pso = pd.read_csv('results/pso_best.csv')

    # Save a summary of final best fitness and position
    summary = pd.DataFrame([
        ['GA', ga.loc[0, 'x'], ga.loc[0, 'y'], ga.loc[0, 'fitness'], ga.loc[0, 'time']],
        ['DE', de.loc[0, 'x'], de.loc[0, 'y'], de.loc[0, 'fitness'], de.loc[0, 'time']],
        ['PSO', pso.loc[0, 'x'], pso.loc[0, 'y'], pso.loc[0, 'fitness'], pso.loc[0, 'time']]
    ], columns=['Algorithm', 'x', 'y', 'Fitness', 'Time (s)'])

    summary.to_csv('results/summary.csv', index=False)
    print(summary)

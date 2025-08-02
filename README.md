# Metaheuristic Optimization of Rastrigin Function

This project implements three population-based metaheuristics — **Genetic Algorithm (GA)**, **Differential Evolution (DE)**, and **Particle Swarm Optimization (PSO)** — to minimize the 2D **Rastrigin function**, a standard multimodal benchmark used in global optimization.

## 🚀 Algorithms
- **GA**: Real-coded with top-50% selection, alpha-blend crossover, and Gaussian mutation.
- **DE**: Classic rand/1 mutation and binomial crossover strategy.
- **PSO**: Inertia-weighted velocity update with personal and global best tracking.

---

## 🔧 Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

## 🧪 How to Run

### Run Sensitivity Analysis (25 configs per algorithm)

```bash
python sensitivity_runner.py
```

- Saves: `results/ga_sensitivity_full.csv`, `de_sensitivity_full.csv`, `pso_sensitivity_full.csv`

### Run Best Configurations + Generate Plot

```bash
python main.py
```

- Outputs:
  - Best solutions in `results/ga_best.csv`, `de_best.csv`, `pso_best.csv`
  - Combined performance plot: `results/convergence_plot.png`
  - Final summary: `results/summary.csv`

---

## 📁 Project Structure

```
.
├── ga_algo.py              # Genetic Algorithm implementation
├── de_algo.py              # Differential Evolution
├── pso_algo.py             # Particle Swarm Optimization
├── rastrigin.py            # Benchmark function
├── sensitivity_runner.py   # Parameter sweeps
├── main.py                 # Final runs and plotting
├── plot_results.py         # Convergence + summary
├── results/                # Output directory
```

---

## 📌 Author

**Amirhossein Rampour**  
Student Number: 100985357  
Ontario Tech University – Advanced Optimization  
Summer 2025

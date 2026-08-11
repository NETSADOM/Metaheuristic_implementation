import os
import numpy as np
import pandas as pd
from benchmarks import BENCHMARKS
from evaluator import Evaluator
from optimizers import (
    random_search,
    random_restart_hill_climbing,
    simulated_annealing,
    evolution_strategy,
    genetic_algorithm
)

def run_experiment():
    algorithms = {
        'RS': random_search,
        'HC': random_restart_hill_climbing,
        'SA': simulated_annealing,
        'ES': evolution_strategy,
        'GA': genetic_algorithm
    }
    
    pilot_seeds = list(range(5))
    final_seeds = list(range(100, 120))
    all_seeds = [('pilot', s) for s in pilot_seeds] + [('final', s) for s in final_seeds]
    
    final_results = []
    convergence_data = []
    
    os.makedirs('results', exist_ok=True)
    
    for func_name, meta in BENCHMARKS.items():
        print(f"Running benchmarks for {func_name}...")
        for alg_name, alg_func in algorithms.items():
            for seed_type, seed in all_seeds:
                np.random.seed(seed)
                
                evaluator = Evaluator(func=meta['func'], bounds=meta['bounds'], budget=20000, log_interval=100)
                
                # Run optimization
                alg_func(evaluator, d=10)
                
                final_results.append({
                    'algorithm': alg_name,
                    'function': func_name,
                    'seed_type': seed_type,
                    'seed': seed,
                    'best_f': evaluator.best_f
                })
                
                evals, best_fs = evaluator.get_history()
                for e, f in zip(evals, best_fs):
                    convergence_data.append({
                        'algorithm': alg_name,
                        'function': func_name,
                        'seed_type': seed_type,
                        'seed': seed,
                        'eval_count': e,
                        'best_f': f
                    })
                    
    df_final = pd.DataFrame(final_results)
    df_final.to_csv('results/final_results.csv', index=False)
    
    df_conv = pd.DataFrame(convergence_data)
    df_conv.to_csv('results/convergence.csv', index=False)
    print("Experiments completed. Results saved to results/ directory.")

if __name__ == '__main__':
    run_experiment()

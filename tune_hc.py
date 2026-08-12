import numpy as np
import pandas as pd
from benchmarks import BENCHMARKS
from evaluator import Evaluator
from optimizers import random_restart_hill_climbing

def tune_hc_neighborhood():
    scales_to_test = [0.05, 0.10, 0.20]
    
    pilot_seeds = list(range(5))
    
    results = []
    
    for scale in scales_to_test:
        print(f"\n--- Testing neighborhood_scale = {scale} ---")
        
        for func_name, meta in BENCHMARKS.items():
            best_f_across_seeds = []
            
            for seed in pilot_seeds:
                np.random.seed(seed)
                evaluator = Evaluator(func=meta['func'], bounds=meta['bounds'], budget=20000, log_interval=100)
                
                random_restart_hill_climbing(evaluator, d=10, neighborhood_scale=scale)
                
                best_f_across_seeds.append(evaluator.best_f)
            
            avg_fitness = np.mean(best_f_across_seeds)
            std_fitness = np.std(best_f_across_seeds)
            
            print(f"{func_name:10s} | Avg Best Fitness: {avg_fitness:.4e} (Std: {std_fitness:.4e})")
            
            results.append({
                'Scale': scale,
                'Function': func_name,
                'Avg_Fitness': avg_fitness,
                'Std_Fitness': std_fitness
            })
            
    df = pd.DataFrame(results)
    
    print("\n=== TUNING SUMMARY ===")
    for func_name in BENCHMARKS.keys():
        func_df = df[df['Function'] == func_name]
        best_row = func_df.loc[func_df['Avg_Fitness'].idxmin()]
        print(f"For {func_name}, the best scale is {best_row['Scale']} (Avg: {best_row['Avg_Fitness']:.4e})")

if __name__ == '__main__':
    tune_hc_neighborhood()

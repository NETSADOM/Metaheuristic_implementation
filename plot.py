import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_results():
    if not os.path.exists('results/convergence.csv') or not os.path.exists('results/final_results.csv'):
        print("Results CSVs not found. Run runner.py first.")
        return
        
    df_conv = pd.read_csv('results/convergence.csv')
    df_final = pd.read_csv('results/final_results.csv')
    
    # Filter for final seeds only as per standard reporting (or plot all, we'll plot final)
    df_conv_final = df_conv[df_conv['seed_type'] == 'final']
    df_final_final = df_final[df_final['seed_type'] == 'final']
    
    functions = df_final['function'].unique()
    
    for func in functions:
        # Convergence Plot
        plt.figure(figsize=(10, 6))
        func_conv = df_conv_final[df_conv_final['function'] == func]
        
        for alg in func_conv['algorithm'].unique():
            alg_data = func_conv[func_conv['algorithm'] == alg]
            
            # Group by eval_count to get median and IQR
            grouped = alg_data.groupby('eval_count')['best_f']
            median = grouped.median()
            q1 = grouped.quantile(0.25)
            q3 = grouped.quantile(0.75)
            evals = median.index
            
            plt.plot(evals, median, label=alg)
            plt.fill_between(evals, q1, q3, alpha=0.2)
            
        plt.title(f'Convergence on {func} (Median ± IQR)')
        plt.xlabel('Evaluations')
        plt.ylabel('Best-so-far Objective')
        plt.yscale('log') # Log scale is usually better for these functions
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(f'results/convergence_{func}.png')
        plt.close()
        
        # Final Boxplot
        plt.figure(figsize=(8, 6))
        func_final = df_final_final[df_final_final['function'] == func]
        
        sns.boxplot(x='algorithm', y='best_f', data=func_final)
        plt.title(f'Final Best Objective on {func}')
        plt.ylabel('Best Objective')
        plt.yscale('log')
        plt.savefig(f'results/boxplot_{func}.png')
        plt.close()
        
    print("Plots saved in results/ directory.")

if __name__ == '__main__':
    plot_results()

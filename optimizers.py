import numpy as np
from boundaries import clip_bounds
from evaluator import BudgetExceededException

def get_random_solution(bounds, d=10):
    lower, upper = bounds
    return np.random.uniform(lower, upper, d)

def random_search(evaluator, d=10):
    try:
        while True:
            x = get_random_solution(evaluator.bounds, d)
            evaluator.evaluate(x)
    except BudgetExceededException:
        pass
    return evaluator.best_x, evaluator.best_f

def random_restart_hill_climbing(evaluator, d=10, neighborhood_scale=0.1, max_restarts=None):
    lower, upper = evaluator.bounds
    scale = (upper - lower) * neighborhood_scale
    
    try:
        while True: # Outer loop for restarts
            current_x = get_random_solution(evaluator.bounds, d)
            current_f = evaluator.evaluate(current_x)
            
            # Local search
            no_improve_count = 0
            while no_improve_count < 100: # Local budget trigger
                # Generate neighbor
                neighbor_x = current_x + np.random.normal(0, scale, d)
                neighbor_x = clip_bounds(neighbor_x, evaluator.bounds)
                
                neighbor_f = evaluator.evaluate(neighbor_x)
                if neighbor_f < current_f:
                    current_x = neighbor_x
                    current_f = neighbor_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
    except BudgetExceededException:
        pass
    return evaluator.best_x, evaluator.best_f

def simulated_annealing(evaluator, d=10, initial_temp=100.0, cooling_rate=0.99, neighborhood_scale=0.1):
    lower, upper = evaluator.bounds
    scale = (upper - lower) * neighborhood_scale
    temp = initial_temp
    
    try:
        current_x = get_random_solution(evaluator.bounds, d)
        current_f = evaluator.evaluate(current_x)
        
        while temp > 1e-8:
            neighbor_x = current_x + np.random.normal(0, scale, d)
            neighbor_x = clip_bounds(neighbor_x, evaluator.bounds)
            neighbor_f = evaluator.evaluate(neighbor_x)
            
            if neighbor_f < current_f or np.random.rand() < np.exp((current_f - neighbor_f) / temp):
                current_x = neighbor_x
                current_f = neighbor_f
                
            # Cool down
            temp *= cooling_rate
            
        # If cooled down before budget ends, just random search or restart.
        # Simple policy: keep searching randomly to use up budget (or restart SA)
        while True:
            current_x = get_random_solution(evaluator.bounds, d)
            evaluator.evaluate(current_x)
            
    except BudgetExceededException:
        pass
    return evaluator.best_x, evaluator.best_f

def evolution_strategy(evaluator, d=10, mu=20, lambda_=100, sigma=0.1, plus_selection=False):
    """
    (mu, lambda) or (mu + lambda) ES
    Truncation selection, Gaussian mutation
    """
    lower, upper = evaluator.bounds
    scale = (upper - lower) * sigma
    
    try:
        # Initialize mu parents
        parents = [get_random_solution(evaluator.bounds, d) for _ in range(mu)]
        parent_fitness = [evaluator.evaluate(p) for p in parents]
        
        while True:
            offspring = []
            offspring_fitness = []
            
            for _ in range(lambda_):
                # Randomly pick a parent to mutate
                parent = parents[np.random.randint(mu)]
                child = parent + np.random.normal(0, scale, d)
                child = clip_bounds(child, evaluator.bounds)
                offspring.append(child)
                offspring_fitness.append(evaluator.evaluate(child))
                
            if plus_selection:
                pool = parents + offspring
                pool_fitness = parent_fitness + offspring_fitness
            else:
                pool = offspring
                pool_fitness = offspring_fitness
                
            # Truncation selection: pick best mu
            indices = np.argsort(pool_fitness)[:mu]
            parents = [pool[i] for i in indices]
            parent_fitness = [pool_fitness[i] for i in indices]
            
    except BudgetExceededException:
        pass
    return evaluator.best_x, evaluator.best_f

def genetic_algorithm(evaluator, d=10, pop_size=100, mutation_rate=0.1, crossover_type='uniform'):
    """
    Tournament selection, crossover, bounded-uniform mutation.
    """
    lower, upper = evaluator.bounds
    
    try:
        # Initialize population
        pop = [get_random_solution(evaluator.bounds, d) for _ in range(pop_size)]
        fitness = [evaluator.evaluate(p) for p in pop]
        
        while True:
            new_pop = []
            for _ in range(pop_size // 2):
                # Tournament selection
                p1_idx = np.random.choice(pop_size, 3, replace=False)
                p2_idx = np.random.choice(pop_size, 3, replace=False)
                p1 = pop[p1_idx[np.argmin([fitness[i] for i in p1_idx])]]
                p2 = pop[p2_idx[np.argmin([fitness[i] for i in p2_idx])]]
                
                # Crossover
                if crossover_type == 'uniform':
                    mask = np.random.rand(d) < 0.5
                    c1 = np.where(mask, p1, p2)
                    c2 = np.where(mask, p2, p1)
                elif crossover_type == 'one-point':
                    pt = np.random.randint(1, d)
                    c1 = np.concatenate((p1[:pt], p2[pt:]))
                    c2 = np.concatenate((p2[:pt], p1[pt:]))
                elif crossover_type == 'blend':
                    alpha = 0.5
                    c1 = p1 + alpha * (p2 - p1)
                    c2 = p2 + alpha * (p1 - p2)
                else:
                    c1, c2 = np.copy(p1), np.copy(p2)
                    
                # Bounded-uniform mutation
                for child in [c1, c2]:
                    if np.random.rand() < mutation_rate:
                        # Mutate random dimension
                        dim = np.random.randint(d)
                        child[dim] = np.random.uniform(lower, upper)
                    new_pop.append(clip_bounds(child, evaluator.bounds))
            
            pop = new_pop
            fitness = [evaluator.evaluate(p) for p in pop]
            
    except BudgetExceededException:
        pass
    return evaluator.best_x, evaluator.best_f

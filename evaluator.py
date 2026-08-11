import numpy as np

class BudgetExceededException(Exception):
    pass

class Evaluator:
    def __init__(self, func, bounds, budget=20000, log_interval=100):
        self.func = func
        self.bounds = bounds
        self.budget = budget
        self.log_interval = log_interval
        
        self.eval_count = 0
        self.best_f = float('inf')
        self.best_x = None
        
        # Log will store (evaluation_count, best_f)
        self.history = []
        
    def evaluate(self, x):
        if self.eval_count >= self.budget:
            raise BudgetExceededException()
            
        f_val = self.func(x)
        self.eval_count += 1
        
        if f_val < self.best_f:
            self.best_f = f_val
            self.best_x = np.copy(x)
            
        # Log every `log_interval` evaluations (e.g., 100, 200, 300...)
        if self.eval_count % self.log_interval == 0:
            self.history.append((self.eval_count, self.best_f))
            
        return f_val

    def get_history(self):
        """Returns arrays of evaluations and best_f values"""
        if not self.history:
            return np.array([]), np.array([])
        history_arr = np.array(self.history)
        return history_arr[:, 0], history_arr[:, 1]

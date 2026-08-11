import numpy as np

def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    d = len(x)
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    # Sum over i from 0 to d-2
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Metadata for each function: (callable, (lower_bound, upper_bound))
BENCHMARKS = {
    'sphere': {
        'func': sphere,
        'bounds': (-5.12, 5.12)
    },
    'rastrigin': {
        'func': rastrigin,
        'bounds': (-5.12, 5.12)
    },
    'rosenbrock': {
        'func': rosenbrock,
        'bounds': (-5.0, 10.0)
    }
}

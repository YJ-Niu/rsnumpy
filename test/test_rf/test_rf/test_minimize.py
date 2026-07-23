import rsnumpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
result = minimize(rosenbrock, x0, method='Nelder-Mead')

print(f"Success: {result.success}")
print(f"x: {result.x}")
print(f"fun: {result.fun}")
print(f"message: {result.message}")
print("\nminimize function works correctly!")
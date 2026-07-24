import rsnumpy as np
import time
from scipy.interpolate import interp1d

print("Testing scipy.interpolate.interp1d...")
x = np.linspace(0, 10, 1000)
y = np.sin(x)
x_new = np.linspace(0, 10, 2000)

start = time.time()
f = interp1d(x, y)
result = f(x_new)
print(f"interp1d completed in {time.time() - start:.3f}s")
print(f"Result shape: {result.shape}")

print("\nTesting multidimensional interp1d...")
y_2d = np.sin(x).reshape(-1, 1) @ np.ones((1, 4))
start = time.time()
f = interp1d(x, y_2d, axis=0)
result_2d = f(x_new)
print(f"interp1d (2D) completed in {time.time() - start:.3f}s")
print(f"Result shape: {result_2d.shape}")

print("\nTesting complex interp1d...")
y_complex = np.exp(1j * x)
start = time.time()
f = interp1d(x, y_complex)
result_complex = f(x_new)
print(f"interp1d (complex) completed in {time.time() - start:.3f}s")
print(f"Result shape: {result_complex.shape}")

print("\nAll tests passed!")

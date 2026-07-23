# noqa: E501


# Legacy functions - provide a simple fmin implementation to avoid import issues
import rsnumpy as np
# import warnings
from scipy._lib._testutils import PytestTester
# from scipy.optimize import minimize


class _MaxFuncCallError(RuntimeError):
    pass

def _wrap_scalar_function_maxfun_validation(function, args, maxfun):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        if ncalls[0] >= maxfun:
            raise _MaxFuncCallError("Too many function calls")
        ncalls[0] += 1
        fx = function(np.copy(x), *(wrapper_args + args))
        if not np.isscalar(fx):
            try:
                fx = float(np.asarray(fx).item())
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper

def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, initial_simplex=None):
    x0 = np.atleast_1d(x0).astype(float)
    N = len(x0)
    
    rho = 1.0
    chi = 2.0
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    
    if initial_simplex is None:
        sim = np.empty((N + 1, N), dtype=float)
        sim[0] = x0
        for k in range(N):
            y = np.copy(x0)
            if np.abs(y[k]) > np.finfo(float).eps:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.atleast_2d(initial_simplex).astype(float)
    
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf
    
    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)
    
    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)
    
    try:
        for k in range(N + 1):
            fsim[k] = func(sim[k])
    except _MaxFuncCallError:
        pass
    finally:
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
    
    iterations = 1
    
    while fcalls[0] < maxfun and iterations < maxiter:
        try:
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xtol and np.max(np.abs(fsim[0] - fsim[1:])) <= ftol):
                break
            
            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            fxr = func(xr)
            doshrink = 0
            
            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                fxe = func(xe)
                
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        fxc = func(xc)
                        
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        fxcc = func(xcc)
                        
                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1
                    
                    if doshrink:
                        sim[1:] = sim[0] + sigma * (sim[1:] - sim[0])
                        for j in one2np1:
                            fsim[j] = func(sim[j])
            
            iterations += 1
        except _MaxFuncCallError:
            pass
        
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        
        if callback is not None:
            callback(sim[0])
    
    x = sim[0]
    
    if disp:
        if fcalls[0] >= maxfun:
            print("Maximum number of function evaluations exceeded.")
        elif iterations >= maxiter:
            print("Maximum number of iterations exceeded.")
        else:
            print("Optimization terminated successfully.")
            print(f"         Current function value: {fsim[0]:.6f}")
            print(f"         Iterations: {iterations}")
            print(f"         Function evaluations: {fcalls[0]}")
    
    return x


# Deprecated namespaces, to be removed in v2.0.0


class OptimizeResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


__all__ = [s for s in dir() if not s.startswith('_')]
test = PytestTester(__name__)
del PytestTester

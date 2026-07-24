import rsnumpy as np


def _cholesky(a, lower=False, overwrite_a=False, clean=True):
    a1 = np.asarray(a)
    
    if a1.ndim == 2:
        try:
            if lower:
                result = np.linalg.cholesky(a1)
            else:
                result = np.linalg.cholesky(a1.T).T
            return result, []
        except np.linalg.LinAlgError:
            return a1, [{'num': 0, 'lapack_info': -1}]
    else:
        batch_shape = a1.shape[:-2]
        # n = a1.shape[-1]
        result = np.empty_like(a1)
        err_lst = []
        
        for idx in np.ndindex(batch_shape):
            try:
                if lower:
                    result[idx] = np.linalg.cholesky(a1[idx])
                else:
                    result[idx] = np.linalg.cholesky(a1[idx].T).T
            except np.linalg.LinAlgError:
                result[idx] = a1[idx]
                err_lst.append({'num': idx, 'lapack_info': -1})
        
        return result, err_lst

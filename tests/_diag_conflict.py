"""临时诊断：conflict_fields 在 pytest 环境下的 ra['mean'] 路径。"""
import rsnumpy as np


def test_diag_conflict_fields():
    ra = np.rec.array(
        [(1, 'abc', 2.3), (2, 'xyz', 4.2), (3, 'wrs', 1.3)],
        names='field, shape, mean')
    print('D TYPE:', type(ra).__name__)
    print('D dtype_obj:', getattr(ra, '_dtype_obj', 'NONE'))
    print('D fields:', getattr(ra, '_fields', 'NONE'))
    print('D struct_dt:', np._struct_dt_of(ra))
    try:
        print('D mean:', ra['mean'])
    except Exception as e:
        print('D mean ERR:', type(e).__name__, e)
    ra.mean = [1.1, 2.2, 3.3]
    print('D after set mean:', ra['mean'])
    assert True

"""临时诊断：from_repr eval 在 pytest 环境下的完整 traceback。"""
import rsnumpy as np


def test_diag_from_repr():
    a = np.array([(1, 'ABC'), (2, "DEF")],
                 dtype=[('foo', int), ('bar', 'S4')])
    recordview = a.view(np.dtype((np.record, a.dtype)))
    r = repr(recordview)
    print('D REPR:', repr(r))
    try:
        rv = eval("np." + r, {'np': np, 'rsnumpy': np})
        print('D OK:', type(rv).__name__)
    except Exception:
        import traceback
        traceback.print_exc()
    assert True
